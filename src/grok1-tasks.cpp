#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"

void grokMulInput(TASK_ARGS) {
    TASK_VARIABLES;
    mulScalar(transformer->x, 78.38367176906169f, transformer->spec->dim, nThreads, threadIndex);
}

void grokRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        memset(xb2, 0, spec->dim * sizeof(float));
        for (uint8_t s = 0; s < spec->nSlices; s++) {
            float* xbv = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, s);
            add(xb2, xbv, spec->dim, 1, 0);
        }
        transformer->rms = rms(xb2, spec->dim);
    }
}

void grokRmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);

    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
}

void grokRmfFfnNormJoin(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
}

void grokMoeRms(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
}

void grokMoeRmsNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsMoe, spec->dim, nThreads, threadIndex);
}

void grokMoeRouter(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    block->moeRouterMm->forward(xb, block->moeRouterProbs, nThreads, threadIndex);
}

void grokMoeRouterSoftmax(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        softmax(block->moeRouterProbs, spec->nExperts);
    }
}

void grokMoeTopk(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        assert(spec->nActiveExperts == 2); // TODO
        uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);

        int best0i = -1;
        int best1i = -1;
        float best0v;
        float best1v;
        for (int i = 0; i < spec->nExperts; i++) {
            float prob = block->moeRouterProbs[i];
            if (best0i == -1 || best0v < prob) {
                if ((best0i != -1 && best1i == -1) || best1v < best0v) {
                    best1v = best0v;
                    best1i = best0i;
                }
                best0i = i;
                best0v = prob;
            } else if (best1i == -1 || best1v < prob) {
                best1i = i;
                best1v = prob;
            }
        }

        indexes[0] = (uint8_t)best0i;
        indexes[1] = (uint8_t)best1i;
    }
}

void grokMoeNormWeights(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
        float* weights = (float*)transformer->buffer->getUnit(TB_UNIT_MOE_WEIGHTS);

        float sum = 0.0;
        int i;
        for (i = 0; i < spec->nActiveExperts; i++) {
            sum += block->moeRouterProbs[indexes[i]];
        }
        for (i = 0; i < spec->nActiveExperts; i++) {
            weights[i] = block->moeRouterProbs[indexes[i]] / sum;
        }
    }
}

void grokQuantizeMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void grokSyncMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_INDEXES);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_WEIGHTS);
}

void grokMoeBlock0(TASK_ARGS) {
    TASK_VARIABLES;

    uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        uint8_t e = indexes[ae];

        float* expertUp = &hb[block->moeUpAndGate0Slice->d0 * ae];
        float* expertGate = &block->expertGate[block->moeUpAndGate0Slice->d0 * ae];

        block->moeUpMm[e]->forward(xb, expertUp, nThreads, threadIndex);
        block->moeGateMm[e]->forward(xb, expertGate, nThreads, threadIndex);
    }
}

void grokMoeBlock1(TASK_ARGS) {
    TASK_VARIABLES;
    float* hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        float* expertUp = &hb[block->moeUpAndGate0Slice->d0 * ae];
        float* expertGate = &block->expertGate[block->moeUpAndGate0Slice->d0 * ae];

        if (spec->hiddenAct == SILU) {
            silu(expertGate, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
        } else if (spec->hiddenAct == GELU) {
            gelu(expertGate, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
        } else {
            assert(false);
        }
        mul(expertUp, expertGate, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
    }
}

void grokQuantizeMoeMul(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
}

void grokSyncMoeMulA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
}

void grokSyncMoeMulRearrange(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0 && spec->nSlices > 1) {
        char* hbq = (char*)transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
        size_t bufferBytes = transformer->buffer->getUnitBytes(TB_SLICED_HB_QUANTIZED);
        size_t bufferSliceBytes = transformer->buffer->getSlicedBytes(TB_SLICED_HB_QUANTIZED);

        size_t moeUpBytes = bufferBytes / spec->nActiveExperts;
        size_t moeUp0SliceBytes = getBatchBytes(spec->bufferFloatType, block->moeUpAndGate0Slice->d0, 1);

        char* buffer = new char[bufferBytes];

        for (int s = 0; s < spec->nSlices; s++) {
            for (int ae = 0; ae < spec->nActiveExperts; ae++) {
                memcpy(&buffer[ae * moeUpBytes + s * moeUp0SliceBytes], &hbq[s * bufferSliceBytes + ae * moeUp0SliceBytes], moeUp0SliceBytes);
            }
        }

        memcpy(hbq, buffer, bufferBytes);
        delete[] buffer;
    }
}

void grokSyncMoeMulB(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
}

void grokMoeBlock2(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);
    char* hbq = (char*)transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
    size_t rowBytes = getBatchBytes(spec->bufferFloatType, spec->hiddenDim, 1);

    uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
    float* weights = (float*)transformer->buffer->getUnit(TB_UNIT_MOE_WEIGHTS);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        uint8_t e = indexes[ae];
        float weight = weights[ae];

        char* expertUp = &hbq[rowBytes * ae];
        float* expertDown = ae == 0 ? xb2 : &block->expertDown[block->moeDown0Slice->d0 * (ae - 1)];

        block->moeDownMm[e]->forward(expertUp, expertDown, nThreads, threadIndex);

        mulScalar(expertDown, weight, block->moeDown0Slice->d0, nThreads, threadIndex);
        if (ae > 0) {
            add(xb2, expertDown, block->moeDown0Slice->d0, nThreads, threadIndex);
        }
    }
}

void grokQuantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
}

void grokSyncMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
}

void grokDequantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);
}

void grokMoeRmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        transformer->rms = rms(xb2, spec->dim);
    }
}

void grokMoeRmsNormFinal(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn2, spec->dim, nThreads, threadIndex);
}

void grokMoeAdd(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
}

void grokFinalize(TASK_ARGS) {
    TASK_VARIABLES;
    transformer->wclsMm->forward(transformer->x, transformer->logits, nThreads, threadIndex);
}

void grokFinalize2(TASK_ARGS) {
    TASK_VARIABLES;
    mulScalar(transformer->logits, 0.5773502691896257f, spec->vocabSize, nThreads, threadIndex);
}

TransformerArch buildGrok1Arch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(TASK_WITH_NAME(sendPos), TASK_TYPE_TRANSFER);
    a.I(TASK_WITH_NAME(grokMulInput), TASK_TYPE_INFERENCE);
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(TASK_WITH_NAME(llamaRmsAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaRmsAttNorm), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaQuantizeRmsAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaSyncRmsAtt), TASK_TYPE_TRANSFER);
        a.I(TASK_WITH_NAME(llamaQkv), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaRope), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaMultiheadAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaQuantizeMultiheadAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaQuantizeAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaSyncAtt), TASK_TYPE_TRANSFER);
        a.I(TASK_WITH_NAME(llamaDequantizeAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokRmfFfn), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokRmfFfnNorm), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokRmfFfnNormJoin), TASK_TYPE_INFERENCE);

        a.I(TASK_WITH_NAME(grokMoeRms), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeRmsNorm), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeRouter), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeRouterSoftmax), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeTopk), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeNormWeights), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokQuantizeMoeInput), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokSyncMoeInput), TASK_TYPE_TRANSFER);
        a.I(TASK_WITH_NAME(grokMoeBlock0), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeBlock1), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokQuantizeMoeMul), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokSyncMoeMulA), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokSyncMoeMulRearrange), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokSyncMoeMulB), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeBlock2), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokQuantizeMoeOutput), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokSyncMoeOutput), TASK_TYPE_TRANSFER);
        a.I(TASK_WITH_NAME(grokDequantizeMoeOutput), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeRmsFinal), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeRmsNormFinal), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(grokMoeAdd), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaNextBlock), TASK_TYPE_INFERENCE);
    }

    a.I(TASK_WITH_NAME(llamaRmsFinal), TASK_TYPE_INFERENCE);
    a.I(TASK_WITH_NAME(llamaRmsFinalNorm), TASK_TYPE_INFERENCE);
    a.I(TASK_WITH_NAME(grokFinalize), TASK_TYPE_INFERENCE);
    a.I(TASK_WITH_NAME(grokFinalize2), TASK_TYPE_INFERENCE);

    // worker

    for (int i = 0; i < spec->nLayers; i++) {
        a.W(TASK_WITH_NAME(llamaSyncRmsAtt), TASK_TYPE_TRANSFER);
        a.W(TASK_WITH_NAME(llamaQkv), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(llamaRope), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(llamaMultiheadAtt), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(llamaQuantizeMultiheadAtt), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(llamaAtt), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(llamaQuantizeAtt), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(llamaSyncAtt), TASK_TYPE_TRANSFER);

        a.W(TASK_WITH_NAME(grokSyncMoeInput), TASK_TYPE_TRANSFER);
        a.W(TASK_WITH_NAME(grokMoeBlock0), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokMoeBlock1), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokQuantizeMoeMul), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokSyncMoeMulA), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokSyncMoeMulB), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokMoeBlock2), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokQuantizeMoeOutput), TASK_TYPE_INFERENCE);
        a.W(TASK_WITH_NAME(grokSyncMoeOutput), TASK_TYPE_TRANSFER);

        a.W(TASK_WITH_NAME(llamaNextBlock), TASK_TYPE_INFERENCE);
    }

    return a;
}
