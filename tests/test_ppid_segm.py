from dcnum import segm


def test_ppid_segm_base_with_thresh():
    scls = segm.get_available_segmenters()["thresh"]
    sthr = scls(thresh=-3)
    assert sthr.key() == "thresh"
    assert sthr.get_ppid() == "thresh:t=-3:cle=1^f=1^clo=2"
