#include "../../socket.hpp"
#include <chrono>
#include <cstring>
#include <cstdio>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>

using namespace std::chrono;

unsigned int packageSizes[] = { 128, 256, 512, 768, 1024, 1280, 1518, 2048, 4096, 8192, 16384, 32768, 65536 };
unsigned int packageSizesCount = sizeof(packageSizes) / sizeof(unsigned int);
unsigned int maPackageSize = packageSizes[packageSizesCount - 1];
unsigned int nAttempts = 5000;
int port = 7721;

void setNonBlocking(int socket) {
    int flags = fcntl(socket, F_GETFL, 0);
    if (fcntl(socket, F_SETFL, flags |= O_NONBLOCK) < 0)
        throw std::runtime_error("Cannot set socket flags");
}

void readUdpSocket(int socket, char* buffer, unsigned int size, struct sockaddr_in* clientAddr, socklen_t* clientAddrLen) {
    while (size > 0) {
        ssize_t s0 = recvfrom(socket, buffer, size, MSG_WAITALL, (struct sockaddr*)clientAddr, clientAddrLen);
        if (s0 < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                continue;
            printf("error read: %s\n", strerror(errno));
            throw std::runtime_error("Cannot read from socket");
        }
        size -= s0;
        buffer += s0;
    }
}

void writeUdpSocket(int socket, char* buffer, unsigned int size, struct sockaddr_in* clientAddr, socklen_t clientAddrLen) {
    while (size > 0) {
        ssize_t s0 = sendto(socket, buffer, size, 0, (const struct sockaddr*)clientAddr, clientAddrLen);
        if (s0 < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                continue;
            printf("error write: %s\n", strerror(errno));
            throw std::runtime_error("Cannot write to socket");
        }
        size -= s0;
        buffer += s0;
    }
}

void server() {
    printf("nAttempts: %d\n", nAttempts);
    char buffer[maPackageSize];

    printf("TCP test\n");

    {
        SocketServer server(port);
        Socket socket = server.accept();
        for (long i = 0; i < packageSizesCount; i++) {
            unsigned int currentPackageSize = packageSizes[i];

            long long totalReadTime = 0;
            long long totalWriteTime = 0;
            long long totalTime = 0; // [us]
            for (long a = 0; a < nAttempts; a++) {
                auto t0 = high_resolution_clock::now();
                socket.read(buffer, currentPackageSize);
                auto t1 = high_resolution_clock::now();
                socket.write(buffer, currentPackageSize);
                auto t2 = high_resolution_clock::now();

                totalReadTime += duration_cast<microseconds>(t1 - t0).count();
                totalWriteTime += duration_cast<microseconds>(t2 - t1).count();
                totalTime += duration_cast<microseconds>(t2 - t0).count();
            }

            double nPingPongs = (1.0 / (totalTime / 1000000.0)) * (double)nAttempts;
            printf("[%6d bytes] write: %5lld us, read: %5lld us, total: %5lld us, nPingPongs: %.2f\n",
                currentPackageSize, totalWriteTime, totalReadTime, totalTime, nPingPongs);
        }
    }

    printf("UDP test\n");

    {
        int serverSocket = ::socket(AF_INET, SOCK_DGRAM, 0);

        struct sockaddr_in serverAddr;
        struct sockaddr_in  clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        memset(&serverAddr, 0, sizeof(serverAddr));
        memset(&clientAddr, 0, sizeof(clientAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY; 
        serverAddr.sin_port = htons(port); 
        setNonBlocking(serverSocket);

        if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
            throw std::runtime_error("Cannot bind socket");

        for (long i = 0; i < packageSizesCount; i++) {
            unsigned int currentPackageSize = packageSizes[i];

            long long totalReadTime = 0;
            long long totalWriteTime = 0;
            long long totalTime = 0; // [us]

            //setsockopt(serverSocket, SOL_SOCKET, SO_RCVBUF, &currentPackageSize, sizeof(currentPackageSize));
            //setsockopt(serverSocket, SOL_SOCKET, SO_SNDBUF, &currentPackageSize, sizeof(currentPackageSize));

            for (long a = 0; a < nAttempts; a++) {
                auto t0 = high_resolution_clock::now();

                readUdpSocket(serverSocket, buffer, currentPackageSize, &clientAddr, &clientAddrLen);

                auto t1 = high_resolution_clock::now();

                writeUdpSocket(serverSocket, buffer, currentPackageSize, &clientAddr, clientAddrLen);

                auto t2 = high_resolution_clock::now();

                totalReadTime += duration_cast<microseconds>(t1 - t0).count();
                totalWriteTime += duration_cast<microseconds>(t2 - t1).count();
                totalTime += duration_cast<microseconds>(t2 - t0).count();
            }

            double nPingPongs = (1.0 / (totalTime / 1000000.0)) * (double)nAttempts;
            printf("[%6d bytes] write: %5lld us, read: %5lld us, total: %5lld us, nPingPongs: %.2f\n",
                currentPackageSize, totalWriteTime, totalReadTime, totalTime, nPingPongs);
        }
    }
}

void client(char* host) {
    char buffer[maPackageSize];
    printf("TCP test\n");

    {
        char** hosts = new char*[1];
        hosts[0] = host;
        int* ports = new int[1];
        ports[0] = port;

        SocketPool* pool = SocketPool::connect(1, hosts, ports);
        pool->setTurbo(true);

        for (long i = 0; i < packageSizesCount; i++) {
            unsigned int currentPackageSize = packageSizes[i];

            long long totalReadTime = 0;
            long long totalWriteTime = 0;
            long long totalTime = 0; // [us]
            for (long a = 0; a < nAttempts; a++) {
                auto t0 = high_resolution_clock::now();
                pool->write(0, buffer, currentPackageSize);
                auto t1 = high_resolution_clock::now();
                pool->read(0, buffer, currentPackageSize);
                auto t2 = high_resolution_clock::now();

                totalWriteTime += duration_cast<microseconds>(t1 - t0).count();
                totalReadTime += duration_cast<microseconds>(t2 - t1).count();
                totalTime += duration_cast<microseconds>(t2 - t0).count();
            }

            printf("[%6d bytes] write: %5lld us, read: %5lld us, total: %5lld us\n",
                currentPackageSize, totalWriteTime, totalReadTime, totalTime);
        }

        delete pool;
        delete[] hosts;
        delete[] ports;
    }

    printf("UDP test\n");

    {
        int clientSocket = ::socket(AF_INET, SOCK_DGRAM, 0);
        struct sockaddr_in serverAddr;
        socklen_t serverAddrLen = sizeof(serverAddr);
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        serverAddr.sin_addr.s_addr = inet_addr(host);
        setNonBlocking(clientSocket);

        for (long i = 0; i < packageSizesCount; i++) {
            unsigned int currentPackageSize = packageSizes[i];

            //setsockopt(clientSocket, SOL_SOCKET, SO_RCVBUF, &currentPackageSize, sizeof(currentPackageSize));
            //setsockopt(clientSocket, SOL_SOCKET, SO_SNDBUF, &currentPackageSize, sizeof(currentPackageSize));

            long long totalReadTime = 0;
            long long totalWriteTime = 0;
            long long totalTime = 0; // [us]
            for (long a = 0; a < nAttempts; a++) {
                auto t0 = high_resolution_clock::now();

                writeUdpSocket(clientSocket, buffer, currentPackageSize, &serverAddr, sizeof(serverAddr));

                auto t1 = high_resolution_clock::now();

                readUdpSocket(clientSocket, buffer, currentPackageSize, &serverAddr, &serverAddrLen);

                auto t2 = high_resolution_clock::now();

                totalWriteTime += duration_cast<microseconds>(t1 - t0).count();
                totalReadTime += duration_cast<microseconds>(t2 - t1).count();
                totalTime += duration_cast<microseconds>(t2 - t0).count();
            }

            printf("[%6d bytes] write: %5lld us, read: %5lld us, total: %5lld us\n",
                currentPackageSize, totalWriteTime, totalReadTime, totalTime);
        }
    }
}

int main(int argc, char *argv[]) {
    initSockets();
    if (argc > 1 && strcmp(argv[1], "server") == 0) {
        server();
    } else if (argc > 2 && strcmp(argv[1], "client") == 0) {
        client(argv[2]);
    } else {
        printf("Invalid arguments\n");
    }
}
