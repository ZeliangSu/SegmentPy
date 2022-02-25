=====================================
CNN Models
=====================================

U-net
-----
The :ref:`U-Net <paper1>`.

======= ===================== ======================
layers  input                 output
======= ===================== ======================
conv1   (B, H, W, 1)          (B, H, W, cs)
conv2   (B, H, W, cs)         (B, H, W, cs)
maxpl1  (B, H, W, cs)         (B, H/2, W/2, cs)
conv3   (B, H/2, W/2, cs)     (B, H/2, W/2, 2cs)
conv4   (B, H/2, W/2, 2cs)    (B, H/2, W/2, 2cs)
...     ...                   ...
======= ===================== ======================

.. Note::
    * B: batch size
    * cls: number of class
    * cs: number of convolution kernel per layers

LRCS-Net
--------
The :ref:`LRCS-Net <paper2>` is a trimed model derived from :ref:`Seg-Net <paper2>`.

**Encoder:**

=========== ===================== ======================
layers      input                 output
=========== ===================== ======================
conv1       (B, H, W, 1)          (B, H, W, cs)
maxpl1      (B, H, W, cs)         (B, H/2, W/2, cs)
conv2       (B, H/2, W/2, cs)     (B, H/2, W/2, 2cs)
maxpl2      (B, H/2, W/2, 2cs)    (B, H/4, W/4, 2cs)
conv3       (B, H/4, W/4, 2cs)    (B, H/4, W/4, 4cs)
maxpl3      (B, H/4, W/4, 4cs)    (B, H/8, W/8 4cs)
conv4sigm   (B, H/8, W/8, 4cs)    (B, H/8, W/8, 4cs)
=========== ===================== ======================

**Decoder:**

=========== ===================== ======================
layers      input                 output
=========== ===================== ======================
conv5a      (B, H/8, W/8, 4cs)    (B, H/8, W/8, 4cs)
conv5b      (B, H/8, W/8, 4cs)    (B, H/8, W/8, 4cs)
up1         (B, H/8, W/8, 4cs)    (B, H/4, W/4, 4cs)
conv6a      (B, H/4, W/4, 4cs)    (B, H/4, W/4, 2cs)
conv6b      (B, H/4, W/4, 2cs)    (B, H/4, W/4, 2cs)
up2         (B, H/4, W/4, 2cs)    (B, H/2, W/2, 2cs)
conv7a      (B, H/2, W/2, 2cs)    (B, H/2, W/2, 2cs)
conv7b      (B, H/2, W/2, 2cs)    (B, H/2, W/2, 2cs)
up3         (B, H/2, W/2, 2cs)    (B, H, W, 2cs) 
conv8a      (B, H, W, 2cs)        (B, H, W, cs)
conv8b      (B, H, W, cs)         (B, H, W, cs)
conv8c      (B, H, W, cs)         (B, H, W, cls)
=========== ===================== ======================

.. Note::
    * B: batch size
    * cls: number of class
    * cs: number of convolution kernel per layers