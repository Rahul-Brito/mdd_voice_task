# first line: 303
@_utils.fill_doc
def compute_multi_epi_mask(epi_imgs, lower_cutoff=0.2, upper_cutoff=0.85,
                           connected=True, opening=2, threshold=0.5,
                           target_affine=None, target_shape=None,
                           exclude_zeros=False, n_jobs=1,
                           memory=None, verbose=0):
    """Compute a common mask for several sessions or subjects
    of :term:`fMRI` data.

    Uses the mask-finding algorithms to extract masks for each session
    or subject, and then keep only the main connected component of the
    a given fraction of the intersection of all the masks.

    Parameters
    ----------
    epi_imgs : :obj:`list` of Niimg-like objects
        See :ref:`extracting_data`.
        A list of arrays, each item being a subject or a session.
        3D and 4D images are accepted.

        .. note::

            If 3D images are given, we suggest to use the mean image
            of each session.

    threshold : :obj:`float`, optional
        The inter-session threshold: the fraction of the
        total number of sessions in for which a :term:`voxel` must be
        in the mask to be kept in the common mask.
        threshold=1 corresponds to keeping the intersection of all
        masks, whereas threshold=0 is the union of all masks.
    %(lower_cutoff)s
        Default=0.2.
    %(upper_cutoff)s
        Default=0.85.
    %(connected)s
        Default=True.
    %(opening)s
        Default=2.
    exclude_zeros : :obj:`bool`, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
        Default=False.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(memory)s
    %(n_jobs)s

    Returns
    -------
    mask : 3D :class:`nibabel.nifti1.Nifti1Image`
        The brain mask.
    """
    if len(epi_imgs) == 0:
        raise TypeError('An empty object - %r - was passed instead of an '
                        'image or a list of images' % epi_imgs)
    masks = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_epi_mask)(epi_img,
                                  lower_cutoff=lower_cutoff,
                                  upper_cutoff=upper_cutoff,
                                  connected=connected,
                                  opening=opening,
                                  exclude_zeros=exclude_zeros,
                                  target_affine=target_affine,
                                  target_shape=target_shape,
                                  memory=memory)
        for epi_img in epi_imgs)

    mask = intersect_masks(masks, connected=connected, threshold=threshold)
    return mask
