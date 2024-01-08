/***********************************************************************
 * audioProcessing.cu
 * 
 * Auteur: Hicheme BEN GAIED
 * 
 * Ce fichier est une partie intégrante du plugin Rotator pour 
 * VCV Rack, et il est spécifiquement dédié au traitement audio via CUDA.
 * Il contient les définitions des kernels CUDA et des fonctions associées 
 * pour le traitement parallèle des signaux audio.
 * 
 * Fonctionnalités Clés:
 * - Kernel CUDA 'processAudio' pour le traitement parallèle des signaux audio.
 *   Chaque thread du GPU traite un élément du buffer audio, permettant un 
 *   traitement efficace et rapide.
 * - Mixage pondéré des signaux d'entrée en fonction des poids attribués, 
 *   permettant une manipulation flexible des différents signaux audio.
 * - La fonction 'runAudioProcessingKernel' gère la configuration et l'exécution 
 *   des kernels CUDA, y compris la gestion des erreurs.
 * 
 * Ce fichier assurer une qualité sonore optimale lors
 * de la spatialisation et des rotations sonores.
 ***********************************************************************/

#include "audioProcessing.h"
#include <cuda_runtime.h>
#include <cstdio>


//combine source sonor pour avoir une transition pour douce et pas de son qui se teleporte ->rq de pierre
__global__ void processAudio(float *inputs[], float *outputBuffer, int bufferSize, float *weights, int numInputs)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // calcul l'index qui determine quel element du bufer le thread doit traiter
    if (index < bufferSize)
    {
        float outputSample = 0.0f;
        for (int i = 0; i < numInputs; ++i)
        {
            // somme des prduits des poids et des sons entree pour que chaque son egale en gros pour pas qu'un son soit plus influent qu'un autre
            outputSample += weights[i] * inputs[i][index];
        }
        // a la fin on a un mixage ponderee
        outputBuffer[index] = outputSample;
    }
}

// parallelisation sur gpu pour melanger plusieurs signaux audio en fct du poids -> rapide efficace et evite perte de son lors des rota
void runAudioProcessingKernel(float *inputs[], float *outputBuffer, int bufferSize, float *weights, int numInputs)
{
    int blockSize = 256;
    int numBlocks = (bufferSize + blockSize - 1) / blockSize; //nombre de bloc a utilise il est calcule en fonction de la taille du buffer
    processAudio<<<numBlocks, blockSize>>>(inputs, outputBuffer, bufferSize, weights, numInputs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Erreur CUDA: %s\n", cudaGetErrorString(err));
    }
}
