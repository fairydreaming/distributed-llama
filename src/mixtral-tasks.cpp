#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
#include "mixtral-tasks.hpp"

TransformerArch buildMixtralArch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(sendPos, "sendPos", TASK_TYPE_TRANSFER);
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
        a.I(TASK_WITH_NAME(llamaMergeAtt), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaRmfFfn), TASK_TYPE_INFERENCE);
        a.I(TASK_WITH_NAME(llamaRmfFfnNorm), TASK_TYPE_INFERENCE);

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
        a.I(TASK_WITH_NAME(grokMoeAdd), TASK_TYPE_INFERENCE);

        a.I(TASK_WITH_NAME(llamaNextBlock), TASK_TYPE_INFERENCE);
    }
    a.I(TASK_WITH_NAME(llamaRmsFinal), TASK_TYPE_INFERENCE);
    a.I(TASK_WITH_NAME(llamaRmsFinalNorm), TASK_TYPE_INFERENCE);
    a.I(TASK_WITH_NAME(llamaFinalize), TASK_TYPE_INFERENCE);

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
