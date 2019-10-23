# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: extra_link_args = -std=c++11
# encoding: utf-8

import collections
from libcpp.vector cimport vector

cdef extern from "rirgen.h":
    cdef vector[vector[double]] gen_rir(
			double c,
			double fs,
			vector[vector[double]] rr,
			vector[double] ss,
			vector[double] LL,
			vector[double] beta_input,
			vector[double] orientation,
			int isHighPassFilter,
			int nDimension,
			int nOrder,
			int nSamples,
			char microphone_type)


def generate_rir(
		room_measures,
		source_position,
		receiver_positions,
		*,
		reverb_time=None,
		beta_coeffs=None,
		float sound_velocity=340,
		float fs=16000,
		orientation=[.0, .0],
		bint is_high_pass_filter=True,
		int n_dim=3,
		int n_order=-1,
		int n_samples=-1,
		mic_type='o'
):
    """ Computes the response of an acoustic source to one or more microphones in a reverberant room using the image method [1,2].

    Room Impulse Response Generator

    Computes the response of an acoustic source to one or more
    microphones in a reverberant room using the image method [1,2].

    Author    : dr.ir. Emanuel Habets (ehabets@dereverberation.org)

    Version   : 2.1.20141124

    Copyright (C) 2003-2014 E.A.P. Habets, The Netherlands.

    [1] J.B. Allen and D.A. Berkley,
            Image method for efficiently simulating small-room acoustics,
            Journal Acoustic Society of America,
            65(4), April 1979, p 943.

    [2] P.M. Peterson,
            Simulating the response of multiple microphones to a single
            acoustic source in a reverberant room, Journal Acoustic
            Society of America, 80(5), November 1986.

    Args:
            c (float): sound velocity in m/s
            fs (float): sampling frequency in Hz
            receiver_positions (list[list[float]]): M x 3 array specifying
            the (x,y,z) coordinates of the receiver(s) in m
            source_position (list[float]): 1 x 3 vector specifying the (x,y,
            z) coordinates of the source in m
            room_measures (list[float]): 1 x 3 vector specifying the room
            dimensions (x,y,z) in m
            beta_coeffs (list[float]): 1 x 6 vector specifying the reflection
            coefficients [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2]
            reverb_time (float): reverberation time (T_60) in seconds
            n_sample (int): number of samples to calculate, default is T_60*fs
            mic_type (str): [omnidirectional, subcardioid, cardioid,
            hypercardioid, bidirectional], default is omnidirectional
            n_order (int): reflection order, default is -1, i.e. maximum order
            n_dim (int): room dimension (2 or 3), default is 3
            orientation (list[float]): direction in which the microphones are pointed, specified using azimuth and elevation angles (in radians), default is [0 0]
            is_high_pass_filter (bool)^: use 'False' to disable high-pass
            filter, the high-pass filter is enabled by default.

    Return:
            list[list[float]]: M x nsample matrix containing the calculated room impulse response(s)
    """
    if not (reverb_time is None) != (beta_coeffs is None):
        raise ValueError('You provide either reverb_time or beta_coeffs.')
    if beta_coeffs is None:
        beta_coeffs = [reverb_time]

    if all(isinstance(e, collections.Iterable) for e in receiver_positions):
        multiple_mics = True
    else:
        multiple_mics = False
        receiver_positions = [receiver_positions]

    h = gen_rir(sound_velocity,
                fs,
                receiver_positions,
                source_position,
                room_measures,
                beta_coeffs,
                orientation,
                is_high_pass_filter,
                n_dim,
                n_order,
                n_samples,
                ord(mic_type[0]))

    if multiple_mics:
        return h
    return h[0]
