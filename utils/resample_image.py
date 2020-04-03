# Simulate motion artifacts module

# library path
import os, sys

lib_path = os.path.abspath('/vol/biomedic/users/aa16914/lib/python2.7/site-packages/')
sys.path.insert(1, lib_path)
lib_path = os.path.abspath(
    '/vol/biomedic/users/aa16914/lib/SimpleITK/build/Wrapping/build/lib.linux-x86_64-2.7/SimpleITK/')
sys.path.insert(1, lib_path)

import SimpleITK as sitk
import numpy as np
import math as mt


def resample_sitk(img, newSpacing, shiftOrigin=(0, 0, 0), interpolator=sitk.sitkBSpline):
    """ This function transforms a nifti image
        Attributes:
            img:            The fixed image that will be transformed (simpleitk type)
            newSpacing:     The translation vector in mm (tx,ty,tz)
            shiftOrigin:    Shift origin by (dx,dy,dz)
            interpolator:   The resampling filter interpolator. For gray images use sitk.sitkBSpline, and for binary images choose sitk.sitkNearestNeighbor
        Return:
            img_resampled:  The resampled image
    """
    T = sitk.Transform(3, sitk.sitkIdentity)

    resizeFilter = sitk.ResampleImageFilter()
    resizeFilter.SetTransform(T)

    oldSize = img.GetSize()
    oldSpacing = img.GetSpacing()

    newSize = (int(oldSize[0] * oldSpacing[0] / newSpacing[0]),
               int(oldSize[1] * oldSpacing[1] / newSpacing[1]),
               int(oldSize[2] * oldSpacing[2] / newSpacing[2]))

    oldOrigin = img.GetOrigin()
    oldDirection = img.GetDirection()

    newOrigin = [x + y for x, y in zip(oldOrigin, shiftOrigin)]
    newDirection = oldDirection

    resizeFilter.SetOutputDirection(newDirection)
    resizeFilter.SetInterpolator(interpolator)
    resizeFilter.SetOutputSpacing(newSpacing)
    resizeFilter.SetOutputOrigin(newOrigin)
    resizeFilter.SetDefaultPixelValue(0)
    resizeFilter.SetSize(newSize)
    # resizeFilter.DebugOn()

    img_resampled = resizeFilter.Execute(img)

    return img_resampled


def transform_affine_sitk(fixed_image_sitk, translation_vector=[0, 0, 0], rotation_vector=[0, 0, 0],
                          scaling_vector=[1, 1, 1], interpolator=sitk.sitkBSpline, spacing=None):
    """ This function transforms a nifti image
        Attributes:
            fixed_image_sitk:       The fixed image that will be transformed (simpleitk type)
            translation_vector:     The translation vector in mm [tx,ty,tz]
            rotation_angels:        The rotation vector that contains the angels in degrees [Rx,Ry,Rz]
            scaling_vector:         The scaling vector [Sx,Sy,Sz]
            interpolator:           The resampling filter interpolator. For gray images use sitk.sitkBSpline, and for binary images choose sitk.sitkNearestNeighbor
        Return:
            moving_image_sitk:      The moving image that has been transformed
    """

    # DefaultPixelValue = fixed_image_sitk.GetPixel(0,0,0)
    size = fixed_image_sitk.GetSize()
    origin = fixed_image_sitk.GetOrigin()
    direction = fixed_image_sitk.GetDirection()

    if spacing is None:
        spacing = fixed_image_sitk.GetSpacing()

    dimension = 3
    Trans = sitk.Transform(dimension, sitk.sitkAffine)

    # Translation
    tx, ty, tz = translation_vector / np.array(spacing)
    dt = np.array([tx, ty, tz, 1])

    # Rotation
    theta_x, theta_y, theta_z = (mt.pi / 180) * np.array(rotation_vector)

    Rx = np.array([
        [1, 0, 0],
        [0, mt.cos(theta_x), -mt.sin(theta_x)],
        [0, mt.sin(theta_x), mt.cos(theta_x)]])
    Ry = np.array([
        [mt.cos(theta_y), 0, mt.sin(theta_y)],
        [0, 1, 0],
        [-mt.sin(theta_y), 0, mt.cos(theta_y)]])
    Rz = np.array([
        [mt.cos(theta_z), -mt.sin(theta_z), 0],
        [mt.sin(theta_z), mt.cos(theta_z), 0],
        [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx.T).T)

    # Scale
    Sx, Sy, Sz = scaling_vector

    R[0, 0] = R[0, 0] / Sx
    R[1, 1] = R[1, 1] / Sy
    R[2, 2] = R[2, 2] / Sz

    # update transformation vector
    trans_vector = np.concatenate((R.flatten(), dt), axis=0)
    Trans.SetParameters(trans_vector)

    # print(Trans)

    # resample filter
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetTransform(Trans)
    resampleFilter.SetOutputDirection(direction)
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetOutputSpacing(spacing)
    resampleFilter.SetOutputOrigin(origin)
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetSize(size)

    # transform the image
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)

    return moving_image_sitk


def transform_skew_sitk(fixed_image_sitk, translation=(0, 0, 0), scale=(1, 1, 1), skew=(0, 0, 0, 0, 0, 0),
                        interpolator=sitk.sitkBSpline, spacing=None):
    """ This function transforms a nifti image
        Attributes:
            fixed_image_sitk:   The fixed image that will be transformed (simpleitk type)
            translation:        The translation vector in mm [tx,ty,tz]
            skew:               The rotation vector that contains the angels in degrees [ax+,ax-,ay+,ay-,az+,az-]
            scale:              The scaling vector [Sx,Sy,Sz]
            interpolator:       The resampling filter interpolator. For gray images use sitk.sitkBSpline, and for binary images choose sitk.sitkNearestNeighbor
        Return:
            moving_image_sitk:      The moving image that has been transformed
    """

    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(fixed_image_sitk)
    min_val = minmax.GetMinimum()
    print('padding value ', min_val)
    # DefaultPixelValue = fixed_image_sitk.GetPixel(0,0,0)
    size = fixed_image_sitk.GetSize()
    origin = fixed_image_sitk.GetOrigin()

    direction = fixed_image_sitk.GetDirection()
    if spacing is None:
        spacing = fixed_image_sitk.GetSpacing()

    dimension = 3

    skew = np.tan(np.radians(np.array(skew)))  # six eqaully spaced values in[0,1], an arbitrary choice
    versor = (0, 0, 0, 1.0)

    skewTransformer = sitk.ScaleSkewVersor3DTransform(scale, skew, versor, translation)

    # print skewTransformer

    # resample filter
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetTransform(skewTransformer)
    resampleFilter.SetOutputDirection(direction)
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetOutputSpacing(spacing)
    resampleFilter.SetOutputOrigin(origin)
    resampleFilter.SetDefaultPixelValue(min_val)
    resampleFilter.SetSize(size)

    # transform the image
    moving_image_sitk = resampleFilter.Execute(fixed_image_sitk)

    return moving_image_sitk


def moveImages(fixed_image_sitk,
               shiftOrigin=(0, 0, 0), newSpacing=(1., 1., 1.),
               interpolator=sitk.sitkBSpline,
               translation_vector_1=[0, 0, 0], translation_vector_2=[0, 0, 0],
               skew_vector_1=(0, 0, 0, 0, 0, 0), skew_vector_2=(0, 0, 0, 0, 0, 0),
               scaling_vector_1=(1, 1, 1), scaling_vector_2=(1, 1, 1)):
    """ This function transforms a nifti image
        Attributes:
            fixed_image_sitk:   The fixed image that will be transformed (simpleitk type)
            shiftOrigin:        Shift origin by (dx,dy,dz)
            newSpacing:         The translation vector in mm (tx,ty,tz)
            interpolator:       The resampling filter interpolator. For gray images use sitk.sitkBSpline, and for binary images choose sitk.sitkNearestNeighbor
            translation:        The translation vector in mm (tx,ty,tz)
            skew:               The rotation vector that contains the angels in degrees [ax+,ax-,ay+,ay-,az+,az-]
            scale:              The scaling vector (Sx,Sy,Sz)
        Return:
            moving_image_sitk0, moving_image0, moving_image1, moving_image2
    """

    # # reample image
    # fixed_image_sitk0  = resample_sitk(fixed_image_sitk, newSpacing=newSpacing, shiftOrigin=shiftOrigin, interpolator=interpolator)
    # fixed_image_sitk1  = resample_sitk(fixed_image_sitk, newSpacing=newSpacing, shiftOrigin=shiftOrigin, interpolator=interpolator)
    # fixed_image_sitk2  = resample_sitk(fixed_image_sitk, newSpacing=newSpacing, shiftOrigin=shiftOrigin, interpolator=interpolator)
    # reample image
    fixed_image_sitk0 = fixed_image_sitk
    fixed_image_sitk1 = fixed_image_sitk
    fixed_image_sitk2 = fixed_image_sitk

    # Transform image
    moving_image_sitk0 = fixed_image_sitk0
    moving_image_sitk1 = transform_skew_sitk(fixed_image_sitk1, translation=translation_vector_1,
                                             scale=scaling_vector_1, skew=skew_vector_1, interpolator=interpolator)
    moving_image_sitk2 = transform_skew_sitk(fixed_image_sitk2, translation=translation_vector_2,
                                             scale=scaling_vector_2, skew=skew_vector_2, interpolator=interpolator)

    moving_image0 = sitk.GetArrayFromImage(moving_image_sitk0)
    moving_image1 = sitk.GetArrayFromImage(moving_image_sitk1)
    moving_image2 = sitk.GetArrayFromImage(moving_image_sitk2)

    return moving_image_sitk0, moving_image0, moving_image1, moving_image2


def generateMotion(fixed_image_sitk, saveSimDir, newSpacing,
                   skew_vector_1, skew_vector_2,
                   translation_vector_1=(0, 0, 0), translation_vector_2=(0, 0, 0),
                   scaling_vector_1=(1, 1, 1), scaling_vector_2=(1, 1, 1)):
    # reample image
    img_array = sitk.GetArrayFromImage(fixed_image_sitk)
    newOrigin = np.array(fixed_image_sitk.GetOrigin())
    shiftOrigin = (0.0, 0.0, 0.0)

    # generate a mask - threshold image between p20 and p100
    p0 = img_array.min().astype('float')
    p20 = np.percentile(img_array, 20)
    print('p0 = ', p0)
    print('p20 = ', p20)
    # p100 = img_array.max().astype('float')

    # threshold_filter = sitk.BinaryThresholdImageFilter()
    # threshold_filter.SetLowerThreshold(p0)
    # threshold_filter.SetUpperThreshold(p100)

    direction = np.reshape(fixed_image_sitk.GetDirection(), (3, 3))
    # direction_ax = direction.ravel()
    # direction_co = direction[[0,2,1]].ravel()
    # direction_sa = direction[[2,1,0]].ravel()
    direction_sign = np.sign(direction)
    direction_ax = direction_sign * abs(direction)
    direction_co = direction_sign[[0, 2, 1]] * abs(direction[[0, 2, 1]])
    direction_sa = direction_sign[[2, 1, 0]] * abs(direction[[2, 1, 0]])

    # fix_coronal_direction
    direction_co[[1, 2]] = -direction_co[[1, 2]]

    direction_ax = direction_ax.ravel()
    direction_co = direction_co.ravel()
    direction_sa = direction_sa.ravel()

    origin_ax = newOrigin
    origin_co = newOrigin
    origin_sa = newOrigin
    # origin_sign = np.sign(newOrigin)
    # origin_ax = origin_sign * abs(newOrigin)
    # origin_co = origin_sign * abs(newOrigin[[0,2,1]])
    # origin_sa = origin_sign * abs(newOrigin[[2,1,0]])

    # direction_ax = (
    # 1.0, 0.0, 0.0,
    # 0.0, 1.0, 0.0,
    # 0.0, 0.0, 1.0)
    #
    # direction_co = (
    #     1.0, 0.0, 0.0,
    #     0.0, 0.0, 1.0,
    #     0.0, 1.0, 0.0)
    #
    # direction_sa = (
    #     0.0, 0.0, 1.0,
    #     0.0, 1.0, 0.0,
    #     1.0, 0.0, 0.0)

    img_ax = img_array.copy()
    img_ax_sitk = sitk.GetImageFromArray(img_ax)
    img_ax_sitk.SetDirection(direction_ax)
    img_ax_sitk.SetSpacing(newSpacing)
    img_ax_sitk.SetOrigin(origin_ax)
    # img_ax_mask = threshold_filter.Execute(img_ax_sitk)
    img_ax_mask = img_ax_sitk >= p0
    sitk.WriteImage(img_ax_sitk, 'org.nii.gz')
    sitk.WriteImage(img_ax_mask, 'mask.nii.gz')

    img_co = np.swapaxes(img_array, 1, 0)
    # img_co = img_array.copy()
    img_co_sitk = sitk.GetImageFromArray(img_co)
    img_co_sitk.SetDirection(direction_co)
    img_co_sitk.SetSpacing(newSpacing)
    img_co_sitk.SetOrigin(origin_co)
    # img_co_mask = threshold_filter.Execute(img_co_sitk)
    img_co_mask = img_co_sitk >= p0

    img_sa = np.swapaxes(img_array, 2, 0)
    img_sa_sitk = sitk.GetImageFromArray(img_sa)
    img_sa_sitk.SetDirection(direction_sa)
    img_sa_sitk.SetSpacing(newSpacing)
    img_sa_sitk.SetOrigin(origin_sa)
    # img_sa_mask = threshold_filter.Execute(img_sa_sitk)
    img_sa_mask = img_sa_sitk >= p0

    ################################################################################################################################################
    ## ---------------------------------------------------- Axial Image ------------------------------------------------------------------------- ##
    ################################################################################################################################################
    # axial-1 (a1) [0,1,2]
    # newSpacing  = (1.25,1.25,2.5)
    # shiftOrigin = (0,0,0)

    moving_image_sitk0, moving_image0, moving_image1, moving_image2 = moveImages(img_ax_sitk, shiftOrigin=shiftOrigin,
                                                                                 newSpacing=newSpacing,
                                                                                 interpolator=sitk.sitkBSpline,
                                                                                 translation_vector_1=translation_vector_1,
                                                                                 translation_vector_2=translation_vector_2,
                                                                                 skew_vector_1=skew_vector_1,
                                                                                 skew_vector_2=skew_vector_2,
                                                                                 scaling_vector_1=scaling_vector_1,
                                                                                 scaling_vector_2=scaling_vector_2)

    # filename = saveSimDir + "moving_image_a1.nii.gz"
    # sitk.WriteImage(moving_image_sitk0,filename)

    moving_mask_sitk0, moving_mask0, moving_mask1, moving_mask2 = moveImages(img_ax_mask, shiftOrigin=shiftOrigin,
                                                                             newSpacing=newSpacing,
                                                                             interpolator=sitk.sitkNearestNeighbor,
                                                                             translation_vector_1=translation_vector_1,
                                                                             translation_vector_2=translation_vector_2,
                                                                             skew_vector_1=skew_vector_1,
                                                                             skew_vector_2=skew_vector_2,
                                                                             scaling_vector_1=scaling_vector_1,
                                                                             scaling_vector_2=scaling_vector_2)

    end = moving_image0.shape[0]

    # sample slices -------------------------------------------------------------------------
    # moving_mask                 = moving_mask0
    # moving_mask[range(0,end,3)] = moving_mask0[range(0,end,3)]
    # moving_mask[range(1,end,3)] = moving_mask1[range(1,end,3)]
    # moving_mask[range(2,end,3)] = moving_mask2[range(2,end,3)]
    # moving_mask_sitk_final      = sitk.GetImageFromArray(np.array(moving_mask,dtype=np.uint8))
    # moving_mask_sitk_final.CopyInformation(moving_mask_sitk0)
    # filename = saveSimDir + "moving_mask_a1.nii.gz"
    # sitk.WriteImage(moving_mask_sitk_final,filename)

    moving_image = moving_image0
    moving_image[range(0, end, 3)] = moving_image0[range(0, end, 3)]
    moving_image[range(1, end, 3)] = moving_image1[range(1, end, 3)]
    moving_image[range(2, end, 3)] = moving_image2[range(2, end, 3)]
    moving_image[moving_image < p20] = p20

    moving_image_sitk_final = sitk.GetImageFromArray(moving_image)
    moving_image_sitk_final.CopyInformation(moving_image_sitk0)
    # moving_image_sitk_final      = sitk.Mask(moving_image_sitk_final, moving_mask_sitk_final, 0)
    filename = saveSimDir + "moving_image_a1.nii.gz"
    moving_image_sitk_final = sitk.RescaleIntensity(moving_image_sitk_final, 0, 1)
    sitk.WriteImage(moving_image_sitk_final, filename)

    ################################################################################################################################################
    ## ---------------------------------------------------- coronal Image ----------------------------------------------------------------------- ##
    ################################################################################################################################################
    # newSpacing  = (1.25,1.25,2.5)
    # shiftOrigin = (0,0,0)

    moving_image_sitk0, moving_image0, moving_image1, moving_image2 = moveImages(img_co_sitk, shiftOrigin=shiftOrigin,
                                                                                 newSpacing=newSpacing,
                                                                                 interpolator=sitk.sitkBSpline,
                                                                                 translation_vector_1=translation_vector_1,
                                                                                 translation_vector_2=translation_vector_2,
                                                                                 skew_vector_1=skew_vector_1,
                                                                                 skew_vector_2=skew_vector_2,
                                                                                 scaling_vector_1=scaling_vector_1,
                                                                                 scaling_vector_2=scaling_vector_2)

    # filename = saveSimDir + "moving_image_c1.nii.gz"
    # sitk.WriteImage(moving_image_sitk0,filename)

    moving_mask_sitk0, moving_mask0, moving_mask1, moving_mask2 = moveImages(img_co_mask, shiftOrigin=shiftOrigin,
                                                                             newSpacing=newSpacing,
                                                                             interpolator=sitk.sitkNearestNeighbor,
                                                                             translation_vector_1=translation_vector_1,
                                                                             translation_vector_2=translation_vector_2,
                                                                             skew_vector_1=skew_vector_1,
                                                                             skew_vector_2=skew_vector_2,
                                                                             scaling_vector_1=scaling_vector_1,
                                                                             scaling_vector_2=scaling_vector_2)

    end = moving_image0.shape[0]

    # sample slices -------------------------------------------------------------------------
    # coronal-1 (c1) [0,1,2]
    # moving_mask                 = moving_mask0
    # moving_mask[range(0,end,3)] = moving_mask0[range(0,end,3)]
    # moving_mask[range(1,end,3)] = moving_mask1[range(1,end,3)]
    # moving_mask[range(2,end,3)] = moving_mask2[range(2,end,3)]
    # moving_mask_sitk_final      = sitk.GetImageFromArray(np.array(moving_mask,dtype=np.uint8))
    # moving_mask_sitk_final.CopyInformation(moving_mask_sitk0)
    # filename = saveSimDir + "moving_mask_c1.nii.gz"
    # sitk.WriteImage(moving_mask_sitk_final,filename)

    moving_image = moving_image0
    moving_image[range(0, end, 3)] = moving_image0[range(0, end, 3)]
    moving_image[range(1, end, 3)] = moving_image1[range(1, end, 3)]
    moving_image[range(2, end, 3)] = moving_image2[range(2, end, 3)]
    moving_image[moving_image < p20] = p20
    moving_image_sitk_final = sitk.GetImageFromArray(moving_image)
    moving_image_sitk_final.CopyInformation(moving_image_sitk0)
    # moving_image_sitk_final      = sitk.Mask(moving_image_sitk_final, moving_mask_sitk_final, 0)
    filename = saveSimDir + "moving_image_c1.nii.gz"
    moving_image_sitk_final = sitk.RescaleIntensity(moving_image_sitk_final, 0, 1)
    sitk.WriteImage(moving_image_sitk_final, filename)

    ################################################################################################################################################
    ## ---------------------------------------------------- saggital Image ---------------------------------------------------------------------- ##
    ################################################################################################################################################

    # newSpacing  = (1.25,1.25,2.5)
    # shiftOrigin = (0,0,0)

    moving_image_sitk0, moving_image0, moving_image1, moving_image2 = moveImages(img_sa_sitk, shiftOrigin=shiftOrigin,
                                                                                 newSpacing=newSpacing,
                                                                                 interpolator=sitk.sitkBSpline,
                                                                                 translation_vector_1=translation_vector_1,
                                                                                 translation_vector_2=translation_vector_2,
                                                                                 skew_vector_1=skew_vector_1,
                                                                                 skew_vector_2=skew_vector_2,
                                                                                 scaling_vector_1=scaling_vector_1,
                                                                                 scaling_vector_2=scaling_vector_2)

    # filename = saveSimDir + "moving_image_s1.nii.gz"
    # sitk.WriteImage(moving_image_sitk0,filename)

    moving_mask_sitk0, moving_mask0, moving_mask1, moving_mask2 = moveImages(img_sa_mask, shiftOrigin=shiftOrigin,
                                                                             newSpacing=newSpacing,
                                                                             interpolator=sitk.sitkNearestNeighbor,
                                                                             translation_vector_1=translation_vector_1,
                                                                             translation_vector_2=translation_vector_2,
                                                                             skew_vector_1=skew_vector_1,
                                                                             skew_vector_2=skew_vector_2,
                                                                             scaling_vector_1=scaling_vector_1,
                                                                             scaling_vector_2=scaling_vector_2)

    end = moving_image0.shape[0]

    # sample slices -------------------------------------------------------------------------
    # saggital-1 (s1) [0,1,2]
    # moving_mask                 = moving_mask0
    # moving_mask[range(0,end,3)] = moving_mask0[range(0,end,3)]
    # moving_mask[range(1,end,3)] = moving_mask1[range(1,end,3)]
    # moving_mask[range(2,end,3)] = moving_mask2[range(2,end,3)]
    # moving_mask_sitk_final      = sitk.GetImageFromArray(np.array(moving_mask,dtype=np.uint8))
    # moving_mask_sitk_final.CopyInformation(moving_mask_sitk0)
    # filename = saveSimDir + "moving_mask_s1.nii.gz"
    # sitk.WriteImage(moving_mask_sitk_final,filename)

    moving_image = moving_image0
    moving_image[range(0, end, 3)] = moving_image0[range(0, end, 3)]
    moving_image[range(1, end, 3)] = moving_image1[range(1, end, 3)]
    moving_image[range(2, end, 3)] = moving_image2[range(2, end, 3)]
    moving_image[moving_image < p20] = p20
    moving_image_sitk_final = sitk.GetImageFromArray(moving_image)
    moving_image_sitk_final.CopyInformation(moving_image_sitk0)
    # moving_image_sitk_final      = sitk.Mask(moving_image_sitk_final, moving_mask_sitk_final, 0)
    filename = saveSimDir + "moving_image_s1.nii.gz"
    moving_image_sitk_final = sitk.RescaleIntensity(moving_image_sitk_final, 0, 1)
    sitk.WriteImage(moving_image_sitk_final, filename)
