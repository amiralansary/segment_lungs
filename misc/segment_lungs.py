#!/usr/bin/python3

import os
import re
import glob
import numpy as np
import SimpleITK as sitk
from psutil import cpu_count
from operator import sub

num_threads = cpu_count(logical=False)
print(' Set number of threads to ', num_threads)
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(num_threads)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)


###############################################################################
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def tryint(s):
    try:
        return int(s)
    except:
        return s


def list_files(dirpath, dirnames):
    curpath = os.getcwd()
    os.chdir(dirpath)
    f = glob.glob(dirnames)
    f.sort(key=alphanum_key)
    os.chdir(curpath)
    return f


###############################################################################
def read_dicom(files):
    """
    read dicom images from directory
    :param files: dir path that contain the dicom files
    :return: simpleitk image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(files)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


###############################################################################
def normalise_image(image_sitk):
    """

    :param image_sitk:
    :return:
    """
    # suppress an pixel less than 20-percentile to be a background and vice versa
    image_array = sitk.GetArrayFromImage(image_sitk)
    pixels = image_array.ravel()
    q20 = np.quantile(pixels, 0.2)
    q90 = np.quantile(pixels, 0.9)
    norm_image = sitk.Clamp(image_sitk, lowerBound=q20, upperBound=q90)
    norm_image = (norm_image - pixels.mean()) / pixels.std()
    return sitk.RescaleIntensity(norm_image)


###############################################################################
def segment_body(image_sitk):
    """

    :param image_sitk:
    :return:
    """
    # select seed point in the background
    seed = image_sitk.GetSize()
    seed = tuple(map(sub, seed, (1, 1, 1)))
    # region growing from the seed point
    seg_con = sitk.ConnectedThreshold(image_sitk, seedList=[seed], lower=-1, upper=100)
    # sitk.WriteImage(seg_con, 'seg_con.nii.gz')
    # some morphological operations to get rid of isolated islands in the background
    vectorRadius = (20, 20, 20)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)
    # sitk.WriteImage(seg_clean, 'seg_clean.nii.gz')
    # reverse background mask values to get the body mask
    body_mask_0 = seg_clean == 0
    # more morphological operations to clean the body mask
    vectorRadius = (3, 3, 3)
    body_mask_0 = sitk.BinaryMorphologicalOpening(body_mask_0, vectorRadius, kernel)
    # sitk.WriteImage(body_mask_0, 'body_mask_0.nii.gz')
    print('Refining body mask...')
    # find biggest connected component, which is supposed to be the body
    body_mask = sitk.ConnectedComponent(body_mask_0)
    # sitk.WriteImage(body_mask, 'body_mask_1.nii.gz')
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(body_mask)
    # filter out smaller components
    label_sizes = [stats.GetNumberOfPixels(l) for l in stats.GetLabels()]
    biggest_labels = np.argsort(label_sizes)[::-1]
    return body_mask == stats.GetLabels()[biggest_labels[0]]  # biggest component has the highest label value


###############################################################################
def segment_lungs(image_stik):
    """

    :param image_stik:
    :return:
    """
    # Binary threshold
    extracted_lungs_0 = sitk.BinaryThreshold(image_stik, lowerThreshold=20., upperThreshold=50.)
    # sitk.WriteImage(extracted_lungs_0, 'extracted_lungs_0.nii.gz')
    # some morphological operations to get rid of isolated islands in the background
    vectorRadius = (5, 5, 5)
    kernel = sitk.sitkBall
    extracted_lungs_1 = sitk.BinaryMorphologicalClosing(extracted_lungs_0, vectorRadius, kernel)
    vectorRadius = (2, 2, 2)
    extracted_lungs_1 = sitk.BinaryMorphologicalOpening(extracted_lungs_1, vectorRadius, kernel)
    # sitk.WriteImage(extracted_lungs_1, 'extracted_lungs_1.nii.gz')
    # find biggest connected component, which is supposed to be the body
    extracted_lungs_2 = sitk.ConnectedComponent(extracted_lungs_1)
    # sitk.WriteImage(extracted_lungs_2, 'extracted_lungs_2.nii.gz')
    # find biggest components
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(extracted_lungs_2)
    # filter out smaller components
    label_sizes = [stats.GetNumberOfPixels(l) for l in stats.GetLabels()]
    biggest_labels = np.argsort(label_sizes)[::-1]
    # biggest two components are the right and left lungs
    right_lung = extracted_lungs_2 == stats.GetLabels()[biggest_labels[0]]
    left_lung = extracted_lungs_2 == stats.GetLabels()[biggest_labels[1]]
    # some morphological operations to get rid of isolated islands in the background
    print('Refining lung masks...')
    left_lung = sitk.BinaryFillhole(left_lung)
    right_lung = sitk.BinaryFillhole(right_lung)
    vectorRadius = (20, 20, 20)
    right_lung = sitk.BinaryMorphologicalClosing(right_lung, vectorRadius, kernel)
    left_lung = sitk.BinaryMorphologicalClosing(left_lung, vectorRadius, kernel)
    vectorRadius = (2, 2, 2)
    right_lung = sitk.BinaryMorphologicalOpening(right_lung, vectorRadius, kernel)
    left_lung = sitk.BinaryMorphologicalOpening(left_lung, vectorRadius, kernel)
    vectorRadius = (20, 20, 20)
    right_lung = sitk.BinaryMorphologicalClosing(right_lung, vectorRadius, kernel)
    left_lung = sitk.BinaryMorphologicalClosing(left_lung, vectorRadius, kernel)
    # dilate the mask 2 pixels to recover the smoothing effect
    right_lung = sitk.BinaryDilate(right_lung, 2, kernel)
    left_lung = sitk.BinaryDilate(left_lung, 2, kernel)
    return right_lung + 2 * left_lung  # return merged labels


###############################################################################
# Read nifti
# data_dir = '/Users/amiralansary/PycharmProjects/covid-19/data/nifti/Positive'
data_dir = '/Users/amiralansary/PycharmProjects/covid-19/data/nifti/Negative'

cases = list_files(data_dir, '*')


for index, case in enumerate(cases):
    print('=' * 20)
    case_path = os.path.join(data_dir, case)
    filename = list_files(case_path, '*.nii.gz')[0]
    image_path = os.path.join(case_path, filename)
    save_path = image_path[:-7]

    print('Processing subject [{}/{}] - {} ...'.format(index+1, len(cases), image_path))

    image_sitk = sitk.ReadImage(image_path)

    print('Normalising...')
    norm_image_sitk = normalise_image(image_sitk)
    sitk.WriteImage(norm_image_sitk, save_path + '_normalised.nii.gz')
    print('Done!')

    print('Smoothing...')
    smooth_image_sitk = sitk.SmoothingRecursiveGaussian(norm_image_sitk, 2.)
    sitk.WriteImage(smooth_image_sitk, save_path + '_smooth2.nii.gz')
    print('Done!')

    print('Segmenting body...')
    body_sitk = segment_body(smooth_image_sitk)
    sitk.WriteImage(body_sitk, save_path + '_body.nii.gz')
    print('Done!')

    print('Segmenting lungs...')
    # mask normalised image to get rid of background
    body_masked_sitk = sitk.Mask(smooth_image_sitk, body_sitk)
    lungs_sitk = segment_lungs(body_masked_sitk)
    sitk.WriteImage(lungs_sitk, save_path + '_lungs.nii.gz')
    print('Done!')
