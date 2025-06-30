from random import randint

import numpy as np



def characteristicMatrix(nrOfPoints):
    nrOfSets = 2**nrOfPoints
    setIndices = np.arange(nrOfSets)
    charMatrix = np.empty((nrOfPoints, nrOfSets), dtype=bool)
    for pointIdx in range(nrOfPoints):
        relativeIndices = setIndices//(2**pointIdx)
        charIndices = relativeIndices % 2
        charMatrix[pointIdx] = charIndices
    return charMatrix


def singleFacetTest(facet, charMatrix, beta):
    resultMatrix = np.dot(charMatrix,facet)
    leftBool = (resultMatrix <= beta)
    rightBool = (resultMatrix >= beta)
    return np.any(leftBool), leftBool, np.any(rightBool), rightBool


def symmBasis(matrixDim):
    basisMatrices = np.zeros((matrixDim*(matrixDim+1)/2, matrixDim, matrixDim))
    basisNr = 0
    for idx1 in range(matrixDim):
        for idx2 in range(idx1, matrixDim):
            basisMatrices[basisNr,idx1,idx2] = 1
            basisMatrices[basisNr, idx2, idx1] = 1
            basisNr += 1
    return basisMatrices.reshape((matrixDim*(matrixDim+1)/2,matrixDim*matrixDim))


if __name__ == '__main__':
    charMat = characteristicMatrix(6)
    bqpVerts = np.zeros((64, 6, 6))
    for idx in range(64):
        bqpVerts[idx] = np.outer(charMat.T[idx], charMat.T[idx])
    bqpVertFlat = bqpVerts.reshape((64, 36))

    basisMat = symmBasis(6)


