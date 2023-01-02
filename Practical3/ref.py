"""
Poisson Image Editing
William Emmanuel
wemmanuel3@gatech.edu
CS 6745 Final Project Fall 2017
"""

import numpy as np
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix

import cv2


class Poission:
    # Helper enum
    OMEGA = 0
    DEL_OMEGA = 1
    OUTSIDE = 2

    # Determine if a given index is inside omega, on the boundary (del omega),
    # or outside the omega region
    @classmethod
    def point_location(cls, index, mask):
        if cls.in_omega(index, mask) == False:
            return cls.OUTSIDE
        if cls.edge(index, mask) == True:
            return cls.DEL_OMEGA
        return cls.OMEGA

    # Determine if a given index is either outside or inside omega
    @staticmethod
    def in_omega(index, mask):
        return mask[index] == 1

    # Deterimine if a given index is on del omega (boundary)
    @classmethod
    def edge(cls, index, mask):
        if cls.in_omega(index, mask) == False:
            return False
        for pt in cls.get_surrounding(index):
            # If the point is inside omega, and a surrounding point is not,
            # then we must be on an edge
            if cls.in_omega(pt, mask) == False:
                return True
        return False

    # Apply the Laplacian operator at a given index
    @staticmethod
    def lapl_at_index(source, index):

        i, j = index
        edge_num = 0
        minus_data = 0

        try:
            temp = source[i+1, j]
            edge_num += 1
            minus_data += temp
        except Exception as e:
            pass

        try:
            temp = source[i-1, j]
            edge_num += 1
            minus_data += temp
        except Exception as e:
            pass

        try:
            temp = source[i, j+1]
            edge_num += 1
            minus_data += temp
        except Exception as e:
            pass

        try:
            temp = source[i, j-1]
            edge_num += 1
            minus_data += temp
        except Exception as e:
            pass

        val = source[i, j] * edge_num - minus_data

        # val = (4 * source[i, j])    \
        #     - (1 * source[i+1, j]) \
        #     - (1 * source[i-1, j]) \
        #     - (1 * source[i, j+1]) \
        #     - (1 * source[i, j-1])
        return val

    # Find the indicies of omega, or where the mask is 1
    @staticmethod
    def mask_indicies(mask):
        nonzero = np.nonzero(mask)
        return nonzero[0], nonzero[1]

    # Get indicies above, below, to the left and right
    @staticmethod
    def get_surrounding(index):
        i, j = index
        return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

    # Create the A sparse matrix
    @classmethod
    def poisson_sparse_matrix(cls, rows, cols):
        # N = number of points in mask
        N = len(list(rows))
        A = lil_matrix((N, N))
        # Set up row for each point in mask

        points = list(zip(rows, cols))
        for i, index in enumerate(points):
            # Should have 4's diagonal
            A[i, i] = 4
            # Get all surrounding points
            for x in cls.get_surrounding(index):
                # If a surrounding point is in the mask, add -1 to index's
                # row at correct position
                if x not in points:
                    continue
                j = points.index(x)
                A[i, j] = -1
        return A

    # Main method
    # Does Poisson image editing on one channel given a source, target, and mask
    @classmethod
    def channel_process(cls, source, target, mask):
        rows, cols = cls.mask_indicies(mask)

        assert len(rows) == len(cols)
        N = len(rows)
        # Create poisson A matrix. Contains mostly 0's, some 4's and -1's
        A = cls.poisson_sparse_matrix(rows, cols)
        # Create B matrix
        b = np.zeros(N)
        points = list(zip(rows, cols))
        for i, index in enumerate(points):
            # Start with left hand side of discrete equation
            b[i] = cls.lapl_at_index(source, index)
            # If on boundry, add in target intensity
            # Creates constraint lapl source = target at boundary
            if cls.point_location(index, mask) == cls.DEL_OMEGA:
                for pt in cls.get_surrounding(index):
                    if cls.in_omega(pt, mask) == False:
                        b[i] += target[pt]

        # Solve for x, unknown intensities
        x = linalg.cg(A, b)
        # Copy target photo, make sure as int
        composite = np.copy(target).astype(int)
        # Place new intensity on target at given index
        for i, index in enumerate(points):
            composite[index] = x[0][i]

        composite = np.clip(composite, 0, 255)
        return composite.astype('uint8')

    @staticmethod
    def gray_channel_3_image(image):
        assert image.ndim == 3
        if (image[:, :, 0] == image[:, :, 1]).all() and (image[:, :, 0] == image[:, :, 2]).all():
            return True
        else:
            return False

    @staticmethod
    def image_resize(img_source, shape=None, factor=None):
        image_H, image_W = img_source.shape[:2]
        if shape is not None:
            return cv2.resize(img_source, shape)

        if factor is not None:
            resized_H = int(round(image_H * factor))
            resized_W = int(round(image_W * factor))
            return cv2.resize(img_source, [resized_W, resized_H])

        else:
            return img_source
