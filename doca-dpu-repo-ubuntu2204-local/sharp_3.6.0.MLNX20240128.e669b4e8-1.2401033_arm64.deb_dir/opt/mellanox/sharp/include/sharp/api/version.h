/**
* Copyright (c) 2001-2016 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See file LICENSE for terms.
*/


#ifndef SHARP_VERSION_H_
#define SHARP_VERSION_H_

#define SHARP_VERNO_MAJOR            3
#define SHARP_VERNO_MINOR            6
#define SHARP_VERNO_REV              "0"
#define SHARP_VERNO_STRING           "3.6"

#define SHARP_MINOR_BIT              (16UL)
#define SHARP_MAJOR_BIT              (24UL)
#define SHARP_API                    ((3L<<SHARP_MAJOR_BIT)|(6L << SHARP_MINOR_BIT))

#define SHARP_VERSION(major, minor)  (((major)<<SHARP_MAJOR_BIT)|((minor)<<SHARP_MINOR_BIT))

#endif
