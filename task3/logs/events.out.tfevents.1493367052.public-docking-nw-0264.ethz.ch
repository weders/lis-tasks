       �K"	   C�@�Abrain.Event:28�K�     �(�	S!C�@�A"��
]
PlaceholderPlaceholder*'
_output_shapes
:���������d*
dtype0*
shape: 
_
Placeholder_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape: 
d
random_normal/shapeConst*
valueB"d      *
_output_shapes
:*
dtype0
W
random_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
Y
random_normal/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
_output_shapes

:d*
dtype0*

seed *
T0*
seed2 
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
_output_shapes

:d*
T0
d
random_normalAddrandom_normal/mulrandom_normal/mean*
_output_shapes

:d*
T0
|
Variable
VariableV2*
_output_shapes

:d*
dtype0*
	container *
shape
:d*
shared_name 
�
Variable/AssignAssignVariablerandom_normal*
use_locking(*
_class
loc:@Variable*
_output_shapes

:d*
validate_shape(*
T0
i
Variable/readIdentityVariable*
_output_shapes

:d*
_class
loc:@Variable*
T0
_
random_normal_1/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Y
random_normal_1/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_1/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
}
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
_output_shapes
:*
T0
f
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes
:*
T0
v

Variable_1
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_1/AssignAssign
Variable_1random_normal_1*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
:*
validate_shape(*
T0
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:*
_class
loc:@Variable_1*
T0
f
random_normal_2/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Y
random_normal_2/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_2/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes

:
*
T0
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes

:
*
T0
~

Variable_2
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_2/AssignAssign
Variable_2random_normal_2*
use_locking(*
_class
loc:@Variable_2*
_output_shapes

:
*
validate_shape(*
T0
o
Variable_2/readIdentity
Variable_2*
_output_shapes

:
*
_class
loc:@Variable_2*
T0
_
random_normal_3/shapeConst*
valueB:
*
_output_shapes
:*
dtype0
Y
random_normal_3/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_3/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
_output_shapes
:
*
dtype0*

seed *
T0*
seed2 
}
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes
:
*
T0
f
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
_output_shapes
:
*
T0
v

Variable_3
VariableV2*
_output_shapes
:
*
dtype0*
	container *
shape:
*
shared_name 
�
Variable_3/AssignAssign
Variable_3random_normal_3*
use_locking(*
_class
loc:@Variable_3*
_output_shapes
:
*
validate_shape(*
T0
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:
*
_class
loc:@Variable_3*
T0
f
random_normal_4/shapeConst*
valueB"
      *
_output_shapes
:*
dtype0
Y
random_normal_4/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_4/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes

:
*
T0
j
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
_output_shapes

:
*
T0
~

Variable_4
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_4/AssignAssign
Variable_4random_normal_4*
use_locking(*
_class
loc:@Variable_4*
_output_shapes

:
*
validate_shape(*
T0
o
Variable_4/readIdentity
Variable_4*
_output_shapes

:
*
_class
loc:@Variable_4*
T0
_
random_normal_5/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Y
random_normal_5/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_5/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes
:*
T0
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
_output_shapes
:*
T0
v

Variable_5
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_5/AssignAssign
Variable_5random_normal_5*
use_locking(*
_class
loc:@Variable_5*
_output_shapes
:*
validate_shape(*
T0
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:*
_class
loc:@Variable_5*
T0
f
random_normal_6/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
Y
random_normal_6/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_6/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
_output_shapes

:*
dtype0*

seed *
T0*
seed2 
�
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
_output_shapes

:*
T0
j
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
_output_shapes

:*
T0
~

Variable_6
VariableV2*
_output_shapes

:*
dtype0*
	container *
shape
:*
shared_name 
�
Variable_6/AssignAssign
Variable_6random_normal_6*
use_locking(*
_class
loc:@Variable_6*
_output_shapes

:*
validate_shape(*
T0
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:*
_class
loc:@Variable_6*
T0
_
random_normal_7/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Y
random_normal_7/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_7/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes
:*
T0
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes
:*
T0
v

Variable_7
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_7/AssignAssign
Variable_7random_normal_7*
use_locking(*
_class
loc:@Variable_7*
_output_shapes
:*
validate_shape(*
T0
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:*
_class
loc:@Variable_7*
T0
f
random_normal_8/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
Y
random_normal_8/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_8/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
_output_shapes

:*
dtype0*

seed *
T0*
seed2 
�
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
_output_shapes

:*
T0
j
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes

:*
T0
~

Variable_8
VariableV2*
_output_shapes

:*
dtype0*
	container *
shape
:*
shared_name 
�
Variable_8/AssignAssign
Variable_8random_normal_8*
use_locking(*
_class
loc:@Variable_8*
_output_shapes

:*
validate_shape(*
T0
o
Variable_8/readIdentity
Variable_8*
_output_shapes

:*
_class
loc:@Variable_8*
T0
_
random_normal_9/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Y
random_normal_9/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_9/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
_output_shapes
:*
T0
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
_output_shapes
:*
T0
v

Variable_9
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_9/AssignAssign
Variable_9random_normal_9*
use_locking(*
_class
loc:@Variable_9*
_output_shapes
:*
validate_shape(*
T0
k
Variable_9/readIdentity
Variable_9*
_output_shapes
:*
_class
loc:@Variable_9*
T0
g
random_normal_10/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Z
random_normal_10/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_10/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
_output_shapes

:
*
T0
m
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
_output_shapes

:
*
T0

Variable_10
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_10/AssignAssignVariable_10random_normal_10*
use_locking(*
_class
loc:@Variable_10*
_output_shapes

:
*
validate_shape(*
T0
r
Variable_10/readIdentityVariable_10*
_output_shapes

:
*
_class
loc:@Variable_10*
T0
`
random_normal_11/shapeConst*
valueB:
*
_output_shapes
:*
dtype0
Z
random_normal_11/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_11/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*
_output_shapes
:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
_output_shapes
:
*
T0
i
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
_output_shapes
:
*
T0
w
Variable_11
VariableV2*
_output_shapes
:
*
dtype0*
	container *
shape:
*
shared_name 
�
Variable_11/AssignAssignVariable_11random_normal_11*
use_locking(*
_class
loc:@Variable_11*
_output_shapes
:
*
validate_shape(*
T0
n
Variable_11/readIdentityVariable_11*
_output_shapes
:
*
_class
loc:@Variable_11*
T0
g
random_normal_12/shapeConst*
valueB"
      *
_output_shapes
:*
dtype0
Z
random_normal_12/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_12/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_12/RandomStandardNormalRandomStandardNormalrandom_normal_12/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_12/mulMul%random_normal_12/RandomStandardNormalrandom_normal_12/stddev*
_output_shapes

:
*
T0
m
random_normal_12Addrandom_normal_12/mulrandom_normal_12/mean*
_output_shapes

:
*
T0

Variable_12
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_12/AssignAssignVariable_12random_normal_12*
use_locking(*
_class
loc:@Variable_12*
_output_shapes

:
*
validate_shape(*
T0
r
Variable_12/readIdentityVariable_12*
_output_shapes

:
*
_class
loc:@Variable_12*
T0
`
random_normal_13/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Z
random_normal_13/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_13/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_13/RandomStandardNormalRandomStandardNormalrandom_normal_13/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
�
random_normal_13/mulMul%random_normal_13/RandomStandardNormalrandom_normal_13/stddev*
_output_shapes
:*
T0
i
random_normal_13Addrandom_normal_13/mulrandom_normal_13/mean*
_output_shapes
:*
T0
w
Variable_13
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_13/AssignAssignVariable_13random_normal_13*
use_locking(*
_class
loc:@Variable_13*
_output_shapes
:*
validate_shape(*
T0
n
Variable_13/readIdentityVariable_13*
_output_shapes
:*
_class
loc:@Variable_13*
T0
g
random_normal_14/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Z
random_normal_14/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_14/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_14/RandomStandardNormalRandomStandardNormalrandom_normal_14/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_14/mulMul%random_normal_14/RandomStandardNormalrandom_normal_14/stddev*
_output_shapes

:
*
T0
m
random_normal_14Addrandom_normal_14/mulrandom_normal_14/mean*
_output_shapes

:
*
T0

Variable_14
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_14/AssignAssignVariable_14random_normal_14*
use_locking(*
_class
loc:@Variable_14*
_output_shapes

:
*
validate_shape(*
T0
r
Variable_14/readIdentityVariable_14*
_output_shapes

:
*
_class
loc:@Variable_14*
T0
`
random_normal_15/shapeConst*
valueB:
*
_output_shapes
:*
dtype0
Z
random_normal_15/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_15/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_15/RandomStandardNormalRandomStandardNormalrandom_normal_15/shape*
_output_shapes
:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_15/mulMul%random_normal_15/RandomStandardNormalrandom_normal_15/stddev*
_output_shapes
:
*
T0
i
random_normal_15Addrandom_normal_15/mulrandom_normal_15/mean*
_output_shapes
:
*
T0
w
Variable_15
VariableV2*
_output_shapes
:
*
dtype0*
	container *
shape:
*
shared_name 
�
Variable_15/AssignAssignVariable_15random_normal_15*
use_locking(*
_class
loc:@Variable_15*
_output_shapes
:
*
validate_shape(*
T0
n
Variable_15/readIdentityVariable_15*
_output_shapes
:
*
_class
loc:@Variable_15*
T0
g
random_normal_16/shapeConst*
valueB"
      *
_output_shapes
:*
dtype0
Z
random_normal_16/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_16/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_16/RandomStandardNormalRandomStandardNormalrandom_normal_16/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_16/mulMul%random_normal_16/RandomStandardNormalrandom_normal_16/stddev*
_output_shapes

:
*
T0
m
random_normal_16Addrandom_normal_16/mulrandom_normal_16/mean*
_output_shapes

:
*
T0

Variable_16
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_16/AssignAssignVariable_16random_normal_16*
use_locking(*
_class
loc:@Variable_16*
_output_shapes

:
*
validate_shape(*
T0
r
Variable_16/readIdentityVariable_16*
_output_shapes

:
*
_class
loc:@Variable_16*
T0
`
random_normal_17/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Z
random_normal_17/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_17/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_17/RandomStandardNormalRandomStandardNormalrandom_normal_17/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
�
random_normal_17/mulMul%random_normal_17/RandomStandardNormalrandom_normal_17/stddev*
_output_shapes
:*
T0
i
random_normal_17Addrandom_normal_17/mulrandom_normal_17/mean*
_output_shapes
:*
T0
w
Variable_17
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_17/AssignAssignVariable_17random_normal_17*
use_locking(*
_class
loc:@Variable_17*
_output_shapes
:*
validate_shape(*
T0
n
Variable_17/readIdentityVariable_17*
_output_shapes
:*
_class
loc:@Variable_17*
T0
g
random_normal_18/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
Z
random_normal_18/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_18/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_18/RandomStandardNormalRandomStandardNormalrandom_normal_18/shape*
_output_shapes

:*
dtype0*

seed *
T0*
seed2 
�
random_normal_18/mulMul%random_normal_18/RandomStandardNormalrandom_normal_18/stddev*
_output_shapes

:*
T0
m
random_normal_18Addrandom_normal_18/mulrandom_normal_18/mean*
_output_shapes

:*
T0

Variable_18
VariableV2*
_output_shapes

:*
dtype0*
	container *
shape
:*
shared_name 
�
Variable_18/AssignAssignVariable_18random_normal_18*
use_locking(*
_class
loc:@Variable_18*
_output_shapes

:*
validate_shape(*
T0
r
Variable_18/readIdentityVariable_18*
_output_shapes

:*
_class
loc:@Variable_18*
T0
`
random_normal_19/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Z
random_normal_19/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_19/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_19/RandomStandardNormalRandomStandardNormalrandom_normal_19/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
�
random_normal_19/mulMul%random_normal_19/RandomStandardNormalrandom_normal_19/stddev*
_output_shapes
:*
T0
i
random_normal_19Addrandom_normal_19/mulrandom_normal_19/mean*
_output_shapes
:*
T0
w
Variable_19
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_19/AssignAssignVariable_19random_normal_19*
use_locking(*
_class
loc:@Variable_19*
_output_shapes
:*
validate_shape(*
T0
n
Variable_19/readIdentityVariable_19*
_output_shapes
:*
_class
loc:@Variable_19*
T0
g
random_normal_20/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
Z
random_normal_20/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_20/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_20/RandomStandardNormalRandomStandardNormalrandom_normal_20/shape*
_output_shapes

:*
dtype0*

seed *
T0*
seed2 
�
random_normal_20/mulMul%random_normal_20/RandomStandardNormalrandom_normal_20/stddev*
_output_shapes

:*
T0
m
random_normal_20Addrandom_normal_20/mulrandom_normal_20/mean*
_output_shapes

:*
T0

Variable_20
VariableV2*
_output_shapes

:*
dtype0*
	container *
shape
:*
shared_name 
�
Variable_20/AssignAssignVariable_20random_normal_20*
use_locking(*
_class
loc:@Variable_20*
_output_shapes

:*
validate_shape(*
T0
r
Variable_20/readIdentityVariable_20*
_output_shapes

:*
_class
loc:@Variable_20*
T0
`
random_normal_21/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Z
random_normal_21/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_21/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_21/RandomStandardNormalRandomStandardNormalrandom_normal_21/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
�
random_normal_21/mulMul%random_normal_21/RandomStandardNormalrandom_normal_21/stddev*
_output_shapes
:*
T0
i
random_normal_21Addrandom_normal_21/mulrandom_normal_21/mean*
_output_shapes
:*
T0
w
Variable_21
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_21/AssignAssignVariable_21random_normal_21*
use_locking(*
_class
loc:@Variable_21*
_output_shapes
:*
validate_shape(*
T0
n
Variable_21/readIdentityVariable_21*
_output_shapes
:*
_class
loc:@Variable_21*
T0
g
random_normal_22/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0
Z
random_normal_22/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_22/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_22/RandomStandardNormalRandomStandardNormalrandom_normal_22/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_22/mulMul%random_normal_22/RandomStandardNormalrandom_normal_22/stddev*
_output_shapes

:
*
T0
m
random_normal_22Addrandom_normal_22/mulrandom_normal_22/mean*
_output_shapes

:
*
T0

Variable_22
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_22/AssignAssignVariable_22random_normal_22*
use_locking(*
_class
loc:@Variable_22*
_output_shapes

:
*
validate_shape(*
T0
r
Variable_22/readIdentityVariable_22*
_output_shapes

:
*
_class
loc:@Variable_22*
T0
`
random_normal_23/shapeConst*
valueB:
*
_output_shapes
:*
dtype0
Z
random_normal_23/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_23/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_23/RandomStandardNormalRandomStandardNormalrandom_normal_23/shape*
_output_shapes
:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_23/mulMul%random_normal_23/RandomStandardNormalrandom_normal_23/stddev*
_output_shapes
:
*
T0
i
random_normal_23Addrandom_normal_23/mulrandom_normal_23/mean*
_output_shapes
:
*
T0
w
Variable_23
VariableV2*
_output_shapes
:
*
dtype0*
	container *
shape:
*
shared_name 
�
Variable_23/AssignAssignVariable_23random_normal_23*
use_locking(*
_class
loc:@Variable_23*
_output_shapes
:
*
validate_shape(*
T0
n
Variable_23/readIdentityVariable_23*
_output_shapes
:
*
_class
loc:@Variable_23*
T0
g
random_normal_24/shapeConst*
valueB"
      *
_output_shapes
:*
dtype0
Z
random_normal_24/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_24/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_24/RandomStandardNormalRandomStandardNormalrandom_normal_24/shape*
_output_shapes

:
*
dtype0*

seed *
T0*
seed2 
�
random_normal_24/mulMul%random_normal_24/RandomStandardNormalrandom_normal_24/stddev*
_output_shapes

:
*
T0
m
random_normal_24Addrandom_normal_24/mulrandom_normal_24/mean*
_output_shapes

:
*
T0

Variable_24
VariableV2*
_output_shapes

:
*
dtype0*
	container *
shape
:
*
shared_name 
�
Variable_24/AssignAssignVariable_24random_normal_24*
use_locking(*
_class
loc:@Variable_24*
_output_shapes

:
*
validate_shape(*
T0
r
Variable_24/readIdentityVariable_24*
_output_shapes

:
*
_class
loc:@Variable_24*
T0
`
random_normal_25/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Z
random_normal_25/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_25/stddevConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
%random_normal_25/RandomStandardNormalRandomStandardNormalrandom_normal_25/shape*
_output_shapes
:*
dtype0*

seed *
T0*
seed2 
�
random_normal_25/mulMul%random_normal_25/RandomStandardNormalrandom_normal_25/stddev*
_output_shapes
:*
T0
i
random_normal_25Addrandom_normal_25/mulrandom_normal_25/mean*
_output_shapes
:*
T0
w
Variable_25
VariableV2*
_output_shapes
:*
dtype0*
	container *
shape:*
shared_name 
�
Variable_25/AssignAssignVariable_25random_normal_25*
use_locking(*
_class
loc:@Variable_25*
_output_shapes
:*
validate_shape(*
T0
n
Variable_25/readIdentityVariable_25*
_output_shapes
:*
_class
loc:@Variable_25*
T0
�
MatMulMatMulPlaceholderVariable/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
U
AddAddMatMulVariable_1/read*'
_output_shapes
:���������*
T0
C
ReluReluAdd*'
_output_shapes
:���������*
T0
�
MatMul_1MatMulReluVariable_2/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
Y
Add_1AddMatMul_1Variable_3/read*'
_output_shapes
:���������
*
T0
G
Relu_1ReluAdd_1*'
_output_shapes
:���������
*
T0
�
MatMul_2MatMulRelu_1Variable_4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Y
Add_2AddMatMul_2Variable_5/read*'
_output_shapes
:���������*
T0
G
Relu_2ReluAdd_2*'
_output_shapes
:���������*
T0
�
MatMul_3MatMulRelu_2Variable_6/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Y
Add_3AddMatMul_3Variable_7/read*'
_output_shapes
:���������*
T0
G
Relu_3ReluAdd_3*'
_output_shapes
:���������*
T0
�
MatMul_4MatMulRelu_3Variable_8/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Y
Add_4AddMatMul_4Variable_9/read*'
_output_shapes
:���������*
T0
G
Relu_4ReluAdd_4*'
_output_shapes
:���������*
T0
�
MatMul_5MatMulRelu_4Variable_10/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
Z
Add_5AddMatMul_5Variable_11/read*'
_output_shapes
:���������
*
T0
G
Relu_5ReluAdd_5*'
_output_shapes
:���������
*
T0
�
MatMul_6MatMulRelu_5Variable_12/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Z
Add_6AddMatMul_6Variable_13/read*'
_output_shapes
:���������*
T0
G
Relu_6ReluAdd_6*'
_output_shapes
:���������*
T0
�
MatMul_7MatMulRelu_6Variable_14/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
Z
Add_7AddMatMul_7Variable_15/read*'
_output_shapes
:���������
*
T0
G
Relu_7ReluAdd_7*'
_output_shapes
:���������
*
T0
�
MatMul_8MatMulRelu_7Variable_16/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Z
Add_8AddMatMul_8Variable_17/read*'
_output_shapes
:���������*
T0
G
Relu_8ReluAdd_8*'
_output_shapes
:���������*
T0
�
MatMul_9MatMulRelu_8Variable_18/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
Z
Add_9AddMatMul_9Variable_19/read*'
_output_shapes
:���������*
T0
G
Relu_9ReluAdd_9*'
_output_shapes
:���������*
T0
�
	MatMul_10MatMulRelu_9Variable_20/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
\
Add_10Add	MatMul_10Variable_21/read*'
_output_shapes
:���������*
T0
I
Relu_10ReluAdd_10*'
_output_shapes
:���������*
T0
�
	MatMul_11MatMulRelu_10Variable_22/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b( *
T0
\
Add_11Add	MatMul_11Variable_23/read*'
_output_shapes
:���������
*
T0
I
Relu_11ReluAdd_11*'
_output_shapes
:���������
*
T0
�
	MatMul_12MatMulRelu_11Variable_24/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
\
Add_12Add	MatMul_12Variable_25/read*'
_output_shapes
:���������*
T0
I
Relu_12ReluAdd_12*'
_output_shapes
:���������*
T0
�
	MatMul_13MatMulRelu_11Variable_24/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
\
Add_13Add	MatMul_13Variable_25/read*'
_output_shapes
:���������*
T0
F
RankConst*
value	B :*
_output_shapes
: *
dtype0
K
ShapeShapeAdd_13*
_output_shapes
:*
out_type0*
T0
H
Rank_1Const*
value	B :*
_output_shapes
: *
dtype0
M
Shape_1ShapeAdd_13*
_output_shapes
:*
out_type0*
T0
G
Sub/yConst*
value	B :*
_output_shapes
: *
dtype0
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
_output_shapes
:*

axis *
N*
T0
T

Slice/sizeConst*
valueB:*
_output_shapes
:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
M
concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
q
concatConcatV2concat/values_0Sliceconcat/axis*
_output_shapes
:*
N*
T0*

Tidx0
k
ReshapeReshapeAdd_13concat*0
_output_shapes
:������������������*
Tshape0*
T0
H
Rank_2Const*
value	B :*
_output_shapes
: *
dtype0
T
Shape_2ShapePlaceholder_1*
_output_shapes
:*
out_type0*
T0
I
Sub_1/yConst*
value	B :*
_output_shapes
: *
dtype0
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
_output_shapes
:*

axis *
N*
T0
V
Slice_1/sizeConst*
valueB:*
_output_shapes
:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
valueB:
���������*
_output_shapes
:*
dtype0
O
concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
_output_shapes
:*
N*
T0*

Tidx0
v
	Reshape_1ReshapePlaceholder_1concat_1*0
_output_shapes
:������������������*
Tshape0*
T0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*?
_output_shapes-
+:���������:������������������*
T0
I
Sub_2/yConst*
value	B :*
_output_shapes
: *
dtype0
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
valueB: *
_output_shapes
:*
dtype0
U
Slice_2/sizePackSub_2*
_output_shapes
:*

axis *
N*
T0
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:���������*
T0*
Index0
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:���������*
Tshape0*
T0
O
ConstConst*
valueB: *
_output_shapes
:*
dtype0
\
MeanMean	Reshape_2Const*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
T
gradients/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients/Mean_grad/ShapeShape	Reshape_2*
_output_shapes
:*
out_type0*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*#
_output_shapes
:���������*
T0
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
_output_shapes
:*
out_type0*
T0
^
gradients/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
c
gradients/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:���������*
T0
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
out_type0*
T0
�
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*#
_output_shapes
:���������*
Tshape0*
T0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*'
_output_shapes
:���������*
T0
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:������������������*
T0
b
gradients/Reshape_grad/ShapeShapeAdd_13*
_output_shapes
:*
out_type0*
T0
�
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
d
gradients/Add_13_grad/ShapeShape	MatMul_13*
_output_shapes
:*
out_type0*
T0
g
gradients/Add_13_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
+gradients/Add_13_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_13_grad/Shapegradients/Add_13_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_13_grad/SumSumgradients/Reshape_grad/Reshape+gradients/Add_13_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_13_grad/ReshapeReshapegradients/Add_13_grad/Sumgradients/Add_13_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_13_grad/Sum_1Sumgradients/Reshape_grad/Reshape-gradients/Add_13_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_13_grad/Reshape_1Reshapegradients/Add_13_grad/Sum_1gradients/Add_13_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
p
&gradients/Add_13_grad/tuple/group_depsNoOp^gradients/Add_13_grad/Reshape ^gradients/Add_13_grad/Reshape_1
�
.gradients/Add_13_grad/tuple/control_dependencyIdentitygradients/Add_13_grad/Reshape'^gradients/Add_13_grad/tuple/group_deps*'
_output_shapes
:���������*0
_class&
$"loc:@gradients/Add_13_grad/Reshape*
T0
�
0gradients/Add_13_grad/tuple/control_dependency_1Identitygradients/Add_13_grad/Reshape_1'^gradients/Add_13_grad/tuple/group_deps*
_output_shapes
:*2
_class(
&$loc:@gradients/Add_13_grad/Reshape_1*
T0
�
gradients/MatMul_13_grad/MatMulMatMul.gradients/Add_13_grad/tuple/control_dependencyVariable_24/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
!gradients/MatMul_13_grad/MatMul_1MatMulRelu_11.gradients/Add_13_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
w
)gradients/MatMul_13_grad/tuple/group_depsNoOp ^gradients/MatMul_13_grad/MatMul"^gradients/MatMul_13_grad/MatMul_1
�
1gradients/MatMul_13_grad/tuple/control_dependencyIdentitygradients/MatMul_13_grad/MatMul*^gradients/MatMul_13_grad/tuple/group_deps*'
_output_shapes
:���������
*2
_class(
&$loc:@gradients/MatMul_13_grad/MatMul*
T0
�
3gradients/MatMul_13_grad/tuple/control_dependency_1Identity!gradients/MatMul_13_grad/MatMul_1*^gradients/MatMul_13_grad/tuple/group_deps*
_output_shapes

:
*4
_class*
(&loc:@gradients/MatMul_13_grad/MatMul_1*
T0
�
gradients/Relu_11_grad/ReluGradReluGrad1gradients/MatMul_13_grad/tuple/control_dependencyRelu_11*'
_output_shapes
:���������
*
T0
d
gradients/Add_11_grad/ShapeShape	MatMul_11*
_output_shapes
:*
out_type0*
T0
g
gradients/Add_11_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
�
+gradients/Add_11_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_11_grad/Shapegradients/Add_11_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_11_grad/SumSumgradients/Relu_11_grad/ReluGrad+gradients/Add_11_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_11_grad/ReshapeReshapegradients/Add_11_grad/Sumgradients/Add_11_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
�
gradients/Add_11_grad/Sum_1Sumgradients/Relu_11_grad/ReluGrad-gradients/Add_11_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_11_grad/Reshape_1Reshapegradients/Add_11_grad/Sum_1gradients/Add_11_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0
p
&gradients/Add_11_grad/tuple/group_depsNoOp^gradients/Add_11_grad/Reshape ^gradients/Add_11_grad/Reshape_1
�
.gradients/Add_11_grad/tuple/control_dependencyIdentitygradients/Add_11_grad/Reshape'^gradients/Add_11_grad/tuple/group_deps*'
_output_shapes
:���������
*0
_class&
$"loc:@gradients/Add_11_grad/Reshape*
T0
�
0gradients/Add_11_grad/tuple/control_dependency_1Identitygradients/Add_11_grad/Reshape_1'^gradients/Add_11_grad/tuple/group_deps*
_output_shapes
:
*2
_class(
&$loc:@gradients/Add_11_grad/Reshape_1*
T0
�
gradients/MatMul_11_grad/MatMulMatMul.gradients/Add_11_grad/tuple/control_dependencyVariable_22/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
!gradients/MatMul_11_grad/MatMul_1MatMulRelu_10.gradients/Add_11_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
w
)gradients/MatMul_11_grad/tuple/group_depsNoOp ^gradients/MatMul_11_grad/MatMul"^gradients/MatMul_11_grad/MatMul_1
�
1gradients/MatMul_11_grad/tuple/control_dependencyIdentitygradients/MatMul_11_grad/MatMul*^gradients/MatMul_11_grad/tuple/group_deps*'
_output_shapes
:���������*2
_class(
&$loc:@gradients/MatMul_11_grad/MatMul*
T0
�
3gradients/MatMul_11_grad/tuple/control_dependency_1Identity!gradients/MatMul_11_grad/MatMul_1*^gradients/MatMul_11_grad/tuple/group_deps*
_output_shapes

:
*4
_class*
(&loc:@gradients/MatMul_11_grad/MatMul_1*
T0
�
gradients/Relu_10_grad/ReluGradReluGrad1gradients/MatMul_11_grad/tuple/control_dependencyRelu_10*'
_output_shapes
:���������*
T0
d
gradients/Add_10_grad/ShapeShape	MatMul_10*
_output_shapes
:*
out_type0*
T0
g
gradients/Add_10_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
+gradients/Add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_10_grad/Shapegradients/Add_10_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_10_grad/SumSumgradients/Relu_10_grad/ReluGrad+gradients/Add_10_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_10_grad/ReshapeReshapegradients/Add_10_grad/Sumgradients/Add_10_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_10_grad/Sum_1Sumgradients/Relu_10_grad/ReluGrad-gradients/Add_10_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_10_grad/Reshape_1Reshapegradients/Add_10_grad/Sum_1gradients/Add_10_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
p
&gradients/Add_10_grad/tuple/group_depsNoOp^gradients/Add_10_grad/Reshape ^gradients/Add_10_grad/Reshape_1
�
.gradients/Add_10_grad/tuple/control_dependencyIdentitygradients/Add_10_grad/Reshape'^gradients/Add_10_grad/tuple/group_deps*'
_output_shapes
:���������*0
_class&
$"loc:@gradients/Add_10_grad/Reshape*
T0
�
0gradients/Add_10_grad/tuple/control_dependency_1Identitygradients/Add_10_grad/Reshape_1'^gradients/Add_10_grad/tuple/group_deps*
_output_shapes
:*2
_class(
&$loc:@gradients/Add_10_grad/Reshape_1*
T0
�
gradients/MatMul_10_grad/MatMulMatMul.gradients/Add_10_grad/tuple/control_dependencyVariable_20/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
!gradients/MatMul_10_grad/MatMul_1MatMulRelu_9.gradients/Add_10_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
w
)gradients/MatMul_10_grad/tuple/group_depsNoOp ^gradients/MatMul_10_grad/MatMul"^gradients/MatMul_10_grad/MatMul_1
�
1gradients/MatMul_10_grad/tuple/control_dependencyIdentitygradients/MatMul_10_grad/MatMul*^gradients/MatMul_10_grad/tuple/group_deps*'
_output_shapes
:���������*2
_class(
&$loc:@gradients/MatMul_10_grad/MatMul*
T0
�
3gradients/MatMul_10_grad/tuple/control_dependency_1Identity!gradients/MatMul_10_grad/MatMul_1*^gradients/MatMul_10_grad/tuple/group_deps*
_output_shapes

:*4
_class*
(&loc:@gradients/MatMul_10_grad/MatMul_1*
T0
�
gradients/Relu_9_grad/ReluGradReluGrad1gradients/MatMul_10_grad/tuple/control_dependencyRelu_9*'
_output_shapes
:���������*
T0
b
gradients/Add_9_grad/ShapeShapeMatMul_9*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_9_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
*gradients/Add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_9_grad/Shapegradients/Add_9_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_9_grad/SumSumgradients/Relu_9_grad/ReluGrad*gradients/Add_9_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_9_grad/ReshapeReshapegradients/Add_9_grad/Sumgradients/Add_9_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_9_grad/Sum_1Sumgradients/Relu_9_grad/ReluGrad,gradients/Add_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_9_grad/Reshape_1Reshapegradients/Add_9_grad/Sum_1gradients/Add_9_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/Add_9_grad/tuple/group_depsNoOp^gradients/Add_9_grad/Reshape^gradients/Add_9_grad/Reshape_1
�
-gradients/Add_9_grad/tuple/control_dependencyIdentitygradients/Add_9_grad/Reshape&^gradients/Add_9_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_9_grad/Reshape*
T0
�
/gradients/Add_9_grad/tuple/control_dependency_1Identitygradients/Add_9_grad/Reshape_1&^gradients/Add_9_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_9_grad/Reshape_1*
T0
�
gradients/MatMul_9_grad/MatMulMatMul-gradients/Add_9_grad/tuple/control_dependencyVariable_18/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_9_grad/MatMul_1MatMulRelu_8-gradients/Add_9_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_9_grad/tuple/group_depsNoOp^gradients/MatMul_9_grad/MatMul!^gradients/MatMul_9_grad/MatMul_1
�
0gradients/MatMul_9_grad/tuple/control_dependencyIdentitygradients/MatMul_9_grad/MatMul)^gradients/MatMul_9_grad/tuple/group_deps*'
_output_shapes
:���������*1
_class'
%#loc:@gradients/MatMul_9_grad/MatMul*
T0
�
2gradients/MatMul_9_grad/tuple/control_dependency_1Identity gradients/MatMul_9_grad/MatMul_1)^gradients/MatMul_9_grad/tuple/group_deps*
_output_shapes

:*3
_class)
'%loc:@gradients/MatMul_9_grad/MatMul_1*
T0
�
gradients/Relu_8_grad/ReluGradReluGrad0gradients/MatMul_9_grad/tuple/control_dependencyRelu_8*'
_output_shapes
:���������*
T0
b
gradients/Add_8_grad/ShapeShapeMatMul_8*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_8_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
*gradients/Add_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_8_grad/Shapegradients/Add_8_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_8_grad/SumSumgradients/Relu_8_grad/ReluGrad*gradients/Add_8_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_8_grad/ReshapeReshapegradients/Add_8_grad/Sumgradients/Add_8_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_8_grad/Sum_1Sumgradients/Relu_8_grad/ReluGrad,gradients/Add_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_8_grad/Reshape_1Reshapegradients/Add_8_grad/Sum_1gradients/Add_8_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/Add_8_grad/tuple/group_depsNoOp^gradients/Add_8_grad/Reshape^gradients/Add_8_grad/Reshape_1
�
-gradients/Add_8_grad/tuple/control_dependencyIdentitygradients/Add_8_grad/Reshape&^gradients/Add_8_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_8_grad/Reshape*
T0
�
/gradients/Add_8_grad/tuple/control_dependency_1Identitygradients/Add_8_grad/Reshape_1&^gradients/Add_8_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_8_grad/Reshape_1*
T0
�
gradients/MatMul_8_grad/MatMulMatMul-gradients/Add_8_grad/tuple/control_dependencyVariable_16/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_8_grad/MatMul_1MatMulRelu_7-gradients/Add_8_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_8_grad/tuple/group_depsNoOp^gradients/MatMul_8_grad/MatMul!^gradients/MatMul_8_grad/MatMul_1
�
0gradients/MatMul_8_grad/tuple/control_dependencyIdentitygradients/MatMul_8_grad/MatMul)^gradients/MatMul_8_grad/tuple/group_deps*'
_output_shapes
:���������
*1
_class'
%#loc:@gradients/MatMul_8_grad/MatMul*
T0
�
2gradients/MatMul_8_grad/tuple/control_dependency_1Identity gradients/MatMul_8_grad/MatMul_1)^gradients/MatMul_8_grad/tuple/group_deps*
_output_shapes

:
*3
_class)
'%loc:@gradients/MatMul_8_grad/MatMul_1*
T0
�
gradients/Relu_7_grad/ReluGradReluGrad0gradients/MatMul_8_grad/tuple/control_dependencyRelu_7*'
_output_shapes
:���������
*
T0
b
gradients/Add_7_grad/ShapeShapeMatMul_7*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_7_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
�
*gradients/Add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_7_grad/Shapegradients/Add_7_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_7_grad/SumSumgradients/Relu_7_grad/ReluGrad*gradients/Add_7_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_7_grad/ReshapeReshapegradients/Add_7_grad/Sumgradients/Add_7_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
�
gradients/Add_7_grad/Sum_1Sumgradients/Relu_7_grad/ReluGrad,gradients/Add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_7_grad/Reshape_1Reshapegradients/Add_7_grad/Sum_1gradients/Add_7_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0
m
%gradients/Add_7_grad/tuple/group_depsNoOp^gradients/Add_7_grad/Reshape^gradients/Add_7_grad/Reshape_1
�
-gradients/Add_7_grad/tuple/control_dependencyIdentitygradients/Add_7_grad/Reshape&^gradients/Add_7_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/Add_7_grad/Reshape*
T0
�
/gradients/Add_7_grad/tuple/control_dependency_1Identitygradients/Add_7_grad/Reshape_1&^gradients/Add_7_grad/tuple/group_deps*
_output_shapes
:
*1
_class'
%#loc:@gradients/Add_7_grad/Reshape_1*
T0
�
gradients/MatMul_7_grad/MatMulMatMul-gradients/Add_7_grad/tuple/control_dependencyVariable_14/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_7_grad/MatMul_1MatMulRelu_6-gradients/Add_7_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_7_grad/tuple/group_depsNoOp^gradients/MatMul_7_grad/MatMul!^gradients/MatMul_7_grad/MatMul_1
�
0gradients/MatMul_7_grad/tuple/control_dependencyIdentitygradients/MatMul_7_grad/MatMul)^gradients/MatMul_7_grad/tuple/group_deps*'
_output_shapes
:���������*1
_class'
%#loc:@gradients/MatMul_7_grad/MatMul*
T0
�
2gradients/MatMul_7_grad/tuple/control_dependency_1Identity gradients/MatMul_7_grad/MatMul_1)^gradients/MatMul_7_grad/tuple/group_deps*
_output_shapes

:
*3
_class)
'%loc:@gradients/MatMul_7_grad/MatMul_1*
T0
�
gradients/Relu_6_grad/ReluGradReluGrad0gradients/MatMul_7_grad/tuple/control_dependencyRelu_6*'
_output_shapes
:���������*
T0
b
gradients/Add_6_grad/ShapeShapeMatMul_6*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_6_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
*gradients/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_6_grad/Shapegradients/Add_6_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_6_grad/SumSumgradients/Relu_6_grad/ReluGrad*gradients/Add_6_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_6_grad/ReshapeReshapegradients/Add_6_grad/Sumgradients/Add_6_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_6_grad/Sum_1Sumgradients/Relu_6_grad/ReluGrad,gradients/Add_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_6_grad/Reshape_1Reshapegradients/Add_6_grad/Sum_1gradients/Add_6_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/Add_6_grad/tuple/group_depsNoOp^gradients/Add_6_grad/Reshape^gradients/Add_6_grad/Reshape_1
�
-gradients/Add_6_grad/tuple/control_dependencyIdentitygradients/Add_6_grad/Reshape&^gradients/Add_6_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_6_grad/Reshape*
T0
�
/gradients/Add_6_grad/tuple/control_dependency_1Identitygradients/Add_6_grad/Reshape_1&^gradients/Add_6_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_6_grad/Reshape_1*
T0
�
gradients/MatMul_6_grad/MatMulMatMul-gradients/Add_6_grad/tuple/control_dependencyVariable_12/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_6_grad/MatMul_1MatMulRelu_5-gradients/Add_6_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_6_grad/tuple/group_depsNoOp^gradients/MatMul_6_grad/MatMul!^gradients/MatMul_6_grad/MatMul_1
�
0gradients/MatMul_6_grad/tuple/control_dependencyIdentitygradients/MatMul_6_grad/MatMul)^gradients/MatMul_6_grad/tuple/group_deps*'
_output_shapes
:���������
*1
_class'
%#loc:@gradients/MatMul_6_grad/MatMul*
T0
�
2gradients/MatMul_6_grad/tuple/control_dependency_1Identity gradients/MatMul_6_grad/MatMul_1)^gradients/MatMul_6_grad/tuple/group_deps*
_output_shapes

:
*3
_class)
'%loc:@gradients/MatMul_6_grad/MatMul_1*
T0
�
gradients/Relu_5_grad/ReluGradReluGrad0gradients/MatMul_6_grad/tuple/control_dependencyRelu_5*'
_output_shapes
:���������
*
T0
b
gradients/Add_5_grad/ShapeShapeMatMul_5*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_5_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
�
*gradients/Add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_5_grad/Shapegradients/Add_5_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_5_grad/SumSumgradients/Relu_5_grad/ReluGrad*gradients/Add_5_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_5_grad/ReshapeReshapegradients/Add_5_grad/Sumgradients/Add_5_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
�
gradients/Add_5_grad/Sum_1Sumgradients/Relu_5_grad/ReluGrad,gradients/Add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_5_grad/Reshape_1Reshapegradients/Add_5_grad/Sum_1gradients/Add_5_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0
m
%gradients/Add_5_grad/tuple/group_depsNoOp^gradients/Add_5_grad/Reshape^gradients/Add_5_grad/Reshape_1
�
-gradients/Add_5_grad/tuple/control_dependencyIdentitygradients/Add_5_grad/Reshape&^gradients/Add_5_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/Add_5_grad/Reshape*
T0
�
/gradients/Add_5_grad/tuple/control_dependency_1Identitygradients/Add_5_grad/Reshape_1&^gradients/Add_5_grad/tuple/group_deps*
_output_shapes
:
*1
_class'
%#loc:@gradients/Add_5_grad/Reshape_1*
T0
�
gradients/MatMul_5_grad/MatMulMatMul-gradients/Add_5_grad/tuple/control_dependencyVariable_10/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_5_grad/MatMul_1MatMulRelu_4-gradients/Add_5_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_5_grad/tuple/group_depsNoOp^gradients/MatMul_5_grad/MatMul!^gradients/MatMul_5_grad/MatMul_1
�
0gradients/MatMul_5_grad/tuple/control_dependencyIdentitygradients/MatMul_5_grad/MatMul)^gradients/MatMul_5_grad/tuple/group_deps*'
_output_shapes
:���������*1
_class'
%#loc:@gradients/MatMul_5_grad/MatMul*
T0
�
2gradients/MatMul_5_grad/tuple/control_dependency_1Identity gradients/MatMul_5_grad/MatMul_1)^gradients/MatMul_5_grad/tuple/group_deps*
_output_shapes

:
*3
_class)
'%loc:@gradients/MatMul_5_grad/MatMul_1*
T0
�
gradients/Relu_4_grad/ReluGradReluGrad0gradients/MatMul_5_grad/tuple/control_dependencyRelu_4*'
_output_shapes
:���������*
T0
b
gradients/Add_4_grad/ShapeShapeMatMul_4*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_4_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
*gradients/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_4_grad/Shapegradients/Add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_4_grad/SumSumgradients/Relu_4_grad/ReluGrad*gradients/Add_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_4_grad/ReshapeReshapegradients/Add_4_grad/Sumgradients/Add_4_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_4_grad/Sum_1Sumgradients/Relu_4_grad/ReluGrad,gradients/Add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_4_grad/Reshape_1Reshapegradients/Add_4_grad/Sum_1gradients/Add_4_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/Add_4_grad/tuple/group_depsNoOp^gradients/Add_4_grad/Reshape^gradients/Add_4_grad/Reshape_1
�
-gradients/Add_4_grad/tuple/control_dependencyIdentitygradients/Add_4_grad/Reshape&^gradients/Add_4_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_4_grad/Reshape*
T0
�
/gradients/Add_4_grad/tuple/control_dependency_1Identitygradients/Add_4_grad/Reshape_1&^gradients/Add_4_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_4_grad/Reshape_1*
T0
�
gradients/MatMul_4_grad/MatMulMatMul-gradients/Add_4_grad/tuple/control_dependencyVariable_8/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_4_grad/MatMul_1MatMulRelu_3-gradients/Add_4_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
�
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*'
_output_shapes
:���������*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul*
T0
�
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*
_output_shapes

:*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1*
T0
�
gradients/Relu_3_grad/ReluGradReluGrad0gradients/MatMul_4_grad/tuple/control_dependencyRelu_3*'
_output_shapes
:���������*
T0
b
gradients/Add_3_grad/ShapeShapeMatMul_3*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_3_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
*gradients/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_3_grad/Shapegradients/Add_3_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/Add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_3_grad/ReshapeReshapegradients/Add_3_grad/Sumgradients/Add_3_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/Add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_3_grad/Reshape_1Reshapegradients/Add_3_grad/Sum_1gradients/Add_3_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/Add_3_grad/tuple/group_depsNoOp^gradients/Add_3_grad/Reshape^gradients/Add_3_grad/Reshape_1
�
-gradients/Add_3_grad/tuple/control_dependencyIdentitygradients/Add_3_grad/Reshape&^gradients/Add_3_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_3_grad/Reshape*
T0
�
/gradients/Add_3_grad/tuple/control_dependency_1Identitygradients/Add_3_grad/Reshape_1&^gradients/Add_3_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_3_grad/Reshape_1*
T0
�
gradients/MatMul_3_grad/MatMulMatMul-gradients/Add_3_grad/tuple/control_dependencyVariable_6/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/Add_3_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
�
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*'
_output_shapes
:���������*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*
T0
�
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
_output_shapes

:*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
T0
�
gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*'
_output_shapes
:���������*
T0
b
gradients/Add_2_grad/ShapeShapeMatMul_2*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_2_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
�
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*'
_output_shapes
:���������*/
_class%
#!loc:@gradients/Add_2_grad/Reshape*
T0
�
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
_output_shapes
:*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
T0
�
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*'
_output_shapes
:���������
*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/Add_2_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:���������
*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*
T0
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
_output_shapes

:
*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
T0
�
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*'
_output_shapes
:���������
*
T0
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
_output_shapes
:*
out_type0*
T0
f
gradients/Add_1_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*'
_output_shapes
:���������
*
Tshape0*
T0
�
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
_output_shapes
:
*
Tshape0*
T0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*'
_output_shapes
:���������
*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
_output_shapes
:
*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/Add_1_grad/tuple/control_dependency*
_output_shapes

:
*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:���������*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes

:
*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:���������*
T0
^
gradients/Add_grad/ShapeShapeMatMul*
_output_shapes
:*
out_type0*
T0
d
gradients/Add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*'
_output_shapes
:���������*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
_output_shapes
:*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0
�
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable/read*'
_output_shapes
:���������d*
transpose_a( *
transpose_b(*
T0
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/Add_grad/tuple/control_dependency*
_output_shapes

:d*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������d*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:d*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0
{
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shape: *
_output_shapes
: *
_class
loc:@Variable*
dtype0*
	container *
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
T0
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
_class
loc:@Variable*
T0
{
beta2_power/initial_valueConst*
valueB
 *w�?*
_class
loc:@Variable*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shape: *
_output_shapes
: *
_class
loc:@Variable*
dtype0*
	container *
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
T0
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Variable/Adam/Initializer/ConstConst*
valueBd*    *
_class
loc:@Variable*
dtype0*
_output_shapes

:d
�
Variable/Adam
VariableV2*
shape
:d*
_output_shapes

:d*
_class
loc:@Variable*
dtype0*
	container *
shared_name 
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable*
_output_shapes

:d*
validate_shape(*
T0
s
Variable/Adam/readIdentityVariable/Adam*
_output_shapes

:d*
_class
loc:@Variable*
T0
�
!Variable/Adam_1/Initializer/ConstConst*
valueBd*    *
_class
loc:@Variable*
dtype0*
_output_shapes

:d
�
Variable/Adam_1
VariableV2*
shape
:d*
_output_shapes

:d*
_class
loc:@Variable*
dtype0*
	container *
shared_name 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable*
_output_shapes

:d*
validate_shape(*
T0
w
Variable/Adam_1/readIdentityVariable/Adam_1*
_output_shapes

:d*
_class
loc:@Variable*
T0
�
!Variable_1/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:
�
Variable_1/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_1*
dtype0*
	container *
shared_name 
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
:*
validate_shape(*
T0
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:*
_class
loc:@Variable_1*
T0
�
#Variable_1/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_1*
dtype0*
_output_shapes
:
�
Variable_1/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_1*
dtype0*
	container *
shared_name 
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
:*
validate_shape(*
T0
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:*
_class
loc:@Variable_1*
T0
�
!Variable_2/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_2*
dtype0*
_output_shapes

:

�
Variable_2/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_2*
dtype0*
	container *
shared_name 
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_2*
_output_shapes

:
*
validate_shape(*
T0
y
Variable_2/Adam/readIdentityVariable_2/Adam*
_output_shapes

:
*
_class
loc:@Variable_2*
T0
�
#Variable_2/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_2*
dtype0*
_output_shapes

:

�
Variable_2/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_2*
dtype0*
	container *
shared_name 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_2*
_output_shapes

:
*
validate_shape(*
T0
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_2*
T0
�
!Variable_3/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

�
Variable_3/Adam
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_3*
dtype0*
	container *
shared_name 
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_3*
_output_shapes
:
*
validate_shape(*
T0
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:
*
_class
loc:@Variable_3*
T0
�
#Variable_3/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_3*
dtype0*
_output_shapes
:

�
Variable_3/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_3*
dtype0*
	container *
shared_name 
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_3*
_output_shapes
:
*
validate_shape(*
T0
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:
*
_class
loc:@Variable_3*
T0
�
!Variable_4/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_4*
dtype0*
_output_shapes

:

�
Variable_4/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_4*
dtype0*
	container *
shared_name 
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_4*
_output_shapes

:
*
validate_shape(*
T0
y
Variable_4/Adam/readIdentityVariable_4/Adam*
_output_shapes

:
*
_class
loc:@Variable_4*
T0
�
#Variable_4/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_4*
dtype0*
_output_shapes

:

�
Variable_4/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_4*
dtype0*
	container *
shared_name 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_4*
_output_shapes

:
*
validate_shape(*
T0
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_4*
T0
�
!Variable_5/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:
�
Variable_5/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_5*
dtype0*
	container *
shared_name 
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_5*
_output_shapes
:*
validate_shape(*
T0
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:*
_class
loc:@Variable_5*
T0
�
#Variable_5/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_5*
dtype0*
_output_shapes
:
�
Variable_5/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_5*
dtype0*
	container *
shared_name 
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_5*
_output_shapes
:*
validate_shape(*
T0
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes
:*
_class
loc:@Variable_5*
T0
�
!Variable_6/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_6*
dtype0*
_output_shapes

:
�
Variable_6/Adam
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_6*
dtype0*
	container *
shared_name 
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_6*
_output_shapes

:*
validate_shape(*
T0
y
Variable_6/Adam/readIdentityVariable_6/Adam*
_output_shapes

:*
_class
loc:@Variable_6*
T0
�
#Variable_6/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_6*
dtype0*
_output_shapes

:
�
Variable_6/Adam_1
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_6*
dtype0*
	container *
shared_name 
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_6*
_output_shapes

:*
validate_shape(*
T0
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_output_shapes

:*
_class
loc:@Variable_6*
T0
�
!Variable_7/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:
�
Variable_7/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_7*
dtype0*
	container *
shared_name 
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_7*
_output_shapes
:*
validate_shape(*
T0
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:*
_class
loc:@Variable_7*
T0
�
#Variable_7/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_7*
dtype0*
_output_shapes
:
�
Variable_7/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_7*
dtype0*
	container *
shared_name 
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_7*
_output_shapes
:*
validate_shape(*
T0
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:*
_class
loc:@Variable_7*
T0
�
!Variable_8/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_8*
dtype0*
_output_shapes

:
�
Variable_8/Adam
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_8*
dtype0*
	container *
shared_name 
�
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_8*
_output_shapes

:*
validate_shape(*
T0
y
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes

:*
_class
loc:@Variable_8*
T0
�
#Variable_8/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_8*
dtype0*
_output_shapes

:
�
Variable_8/Adam_1
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_8*
dtype0*
	container *
shared_name 
�
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_8*
_output_shapes

:*
validate_shape(*
T0
}
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes

:*
_class
loc:@Variable_8*
T0
�
!Variable_9/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:
�
Variable_9/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_9*
dtype0*
	container *
shared_name 
�
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_9*
_output_shapes
:*
validate_shape(*
T0
u
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
#Variable_9/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_9*
dtype0*
_output_shapes
:
�
Variable_9/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_9*
dtype0*
	container *
shared_name 
�
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_9*
_output_shapes
:*
validate_shape(*
T0
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes
:*
_class
loc:@Variable_9*
T0
�
"Variable_10/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_10*
dtype0*
_output_shapes

:

�
Variable_10/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_10*
dtype0*
	container *
shared_name 
�
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_10*
_output_shapes

:
*
validate_shape(*
T0
|
Variable_10/Adam/readIdentityVariable_10/Adam*
_output_shapes

:
*
_class
loc:@Variable_10*
T0
�
$Variable_10/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_10*
dtype0*
_output_shapes

:

�
Variable_10/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_10*
dtype0*
	container *
shared_name 
�
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_10*
_output_shapes

:
*
validate_shape(*
T0
�
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_10*
T0
�
"Variable_11/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

�
Variable_11/Adam
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_11*
dtype0*
	container *
shared_name 
�
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_11*
_output_shapes
:
*
validate_shape(*
T0
x
Variable_11/Adam/readIdentityVariable_11/Adam*
_output_shapes
:
*
_class
loc:@Variable_11*
T0
�
$Variable_11/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_11*
dtype0*
_output_shapes
:

�
Variable_11/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_11*
dtype0*
	container *
shared_name 
�
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_11*
_output_shapes
:
*
validate_shape(*
T0
|
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
_output_shapes
:
*
_class
loc:@Variable_11*
T0
�
"Variable_12/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_12*
dtype0*
_output_shapes

:

�
Variable_12/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_12*
dtype0*
	container *
shared_name 
�
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_12*
_output_shapes

:
*
validate_shape(*
T0
|
Variable_12/Adam/readIdentityVariable_12/Adam*
_output_shapes

:
*
_class
loc:@Variable_12*
T0
�
$Variable_12/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_12*
dtype0*
_output_shapes

:

�
Variable_12/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_12*
dtype0*
	container *
shared_name 
�
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_12*
_output_shapes

:
*
validate_shape(*
T0
�
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_12*
T0
�
"Variable_13/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
:
�
Variable_13/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_13*
dtype0*
	container *
shared_name 
�
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_13*
_output_shapes
:*
validate_shape(*
T0
x
Variable_13/Adam/readIdentityVariable_13/Adam*
_output_shapes
:*
_class
loc:@Variable_13*
T0
�
$Variable_13/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_13*
dtype0*
_output_shapes
:
�
Variable_13/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_13*
dtype0*
	container *
shared_name 
�
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_13*
_output_shapes
:*
validate_shape(*
T0
|
Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
_output_shapes
:*
_class
loc:@Variable_13*
T0
�
"Variable_14/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_14*
dtype0*
_output_shapes

:

�
Variable_14/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_14*
dtype0*
	container *
shared_name 
�
Variable_14/Adam/AssignAssignVariable_14/Adam"Variable_14/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_14*
_output_shapes

:
*
validate_shape(*
T0
|
Variable_14/Adam/readIdentityVariable_14/Adam*
_output_shapes

:
*
_class
loc:@Variable_14*
T0
�
$Variable_14/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_14*
dtype0*
_output_shapes

:

�
Variable_14/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_14*
dtype0*
	container *
shared_name 
�
Variable_14/Adam_1/AssignAssignVariable_14/Adam_1$Variable_14/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_14*
_output_shapes

:
*
validate_shape(*
T0
�
Variable_14/Adam_1/readIdentityVariable_14/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_14*
T0
�
"Variable_15/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

�
Variable_15/Adam
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_15*
dtype0*
	container *
shared_name 
�
Variable_15/Adam/AssignAssignVariable_15/Adam"Variable_15/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_15*
_output_shapes
:
*
validate_shape(*
T0
x
Variable_15/Adam/readIdentityVariable_15/Adam*
_output_shapes
:
*
_class
loc:@Variable_15*
T0
�
$Variable_15/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_15*
dtype0*
_output_shapes
:

�
Variable_15/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_15*
dtype0*
	container *
shared_name 
�
Variable_15/Adam_1/AssignAssignVariable_15/Adam_1$Variable_15/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_15*
_output_shapes
:
*
validate_shape(*
T0
|
Variable_15/Adam_1/readIdentityVariable_15/Adam_1*
_output_shapes
:
*
_class
loc:@Variable_15*
T0
�
"Variable_16/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_16*
dtype0*
_output_shapes

:

�
Variable_16/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_16*
dtype0*
	container *
shared_name 
�
Variable_16/Adam/AssignAssignVariable_16/Adam"Variable_16/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_16*
_output_shapes

:
*
validate_shape(*
T0
|
Variable_16/Adam/readIdentityVariable_16/Adam*
_output_shapes

:
*
_class
loc:@Variable_16*
T0
�
$Variable_16/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_16*
dtype0*
_output_shapes

:

�
Variable_16/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_16*
dtype0*
	container *
shared_name 
�
Variable_16/Adam_1/AssignAssignVariable_16/Adam_1$Variable_16/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_16*
_output_shapes

:
*
validate_shape(*
T0
�
Variable_16/Adam_1/readIdentityVariable_16/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_16*
T0
�
"Variable_17/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
:
�
Variable_17/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_17*
dtype0*
	container *
shared_name 
�
Variable_17/Adam/AssignAssignVariable_17/Adam"Variable_17/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_17*
_output_shapes
:*
validate_shape(*
T0
x
Variable_17/Adam/readIdentityVariable_17/Adam*
_output_shapes
:*
_class
loc:@Variable_17*
T0
�
$Variable_17/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_17*
dtype0*
_output_shapes
:
�
Variable_17/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_17*
dtype0*
	container *
shared_name 
�
Variable_17/Adam_1/AssignAssignVariable_17/Adam_1$Variable_17/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_17*
_output_shapes
:*
validate_shape(*
T0
|
Variable_17/Adam_1/readIdentityVariable_17/Adam_1*
_output_shapes
:*
_class
loc:@Variable_17*
T0
�
"Variable_18/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_18*
dtype0*
_output_shapes

:
�
Variable_18/Adam
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_18*
dtype0*
	container *
shared_name 
�
Variable_18/Adam/AssignAssignVariable_18/Adam"Variable_18/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_18*
_output_shapes

:*
validate_shape(*
T0
|
Variable_18/Adam/readIdentityVariable_18/Adam*
_output_shapes

:*
_class
loc:@Variable_18*
T0
�
$Variable_18/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_18*
dtype0*
_output_shapes

:
�
Variable_18/Adam_1
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_18*
dtype0*
	container *
shared_name 
�
Variable_18/Adam_1/AssignAssignVariable_18/Adam_1$Variable_18/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_18*
_output_shapes

:*
validate_shape(*
T0
�
Variable_18/Adam_1/readIdentityVariable_18/Adam_1*
_output_shapes

:*
_class
loc:@Variable_18*
T0
�
"Variable_19/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:
�
Variable_19/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_19*
dtype0*
	container *
shared_name 
�
Variable_19/Adam/AssignAssignVariable_19/Adam"Variable_19/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_19*
_output_shapes
:*
validate_shape(*
T0
x
Variable_19/Adam/readIdentityVariable_19/Adam*
_output_shapes
:*
_class
loc:@Variable_19*
T0
�
$Variable_19/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_19*
dtype0*
_output_shapes
:
�
Variable_19/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_19*
dtype0*
	container *
shared_name 
�
Variable_19/Adam_1/AssignAssignVariable_19/Adam_1$Variable_19/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_19*
_output_shapes
:*
validate_shape(*
T0
|
Variable_19/Adam_1/readIdentityVariable_19/Adam_1*
_output_shapes
:*
_class
loc:@Variable_19*
T0
�
"Variable_20/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_20*
dtype0*
_output_shapes

:
�
Variable_20/Adam
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_20*
dtype0*
	container *
shared_name 
�
Variable_20/Adam/AssignAssignVariable_20/Adam"Variable_20/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_20*
_output_shapes

:*
validate_shape(*
T0
|
Variable_20/Adam/readIdentityVariable_20/Adam*
_output_shapes

:*
_class
loc:@Variable_20*
T0
�
$Variable_20/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_20*
dtype0*
_output_shapes

:
�
Variable_20/Adam_1
VariableV2*
shape
:*
_output_shapes

:*
_class
loc:@Variable_20*
dtype0*
	container *
shared_name 
�
Variable_20/Adam_1/AssignAssignVariable_20/Adam_1$Variable_20/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_20*
_output_shapes

:*
validate_shape(*
T0
�
Variable_20/Adam_1/readIdentityVariable_20/Adam_1*
_output_shapes

:*
_class
loc:@Variable_20*
T0
�
"Variable_21/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
:
�
Variable_21/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_21*
dtype0*
	container *
shared_name 
�
Variable_21/Adam/AssignAssignVariable_21/Adam"Variable_21/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_21*
_output_shapes
:*
validate_shape(*
T0
x
Variable_21/Adam/readIdentityVariable_21/Adam*
_output_shapes
:*
_class
loc:@Variable_21*
T0
�
$Variable_21/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_21*
dtype0*
_output_shapes
:
�
Variable_21/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_21*
dtype0*
	container *
shared_name 
�
Variable_21/Adam_1/AssignAssignVariable_21/Adam_1$Variable_21/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_21*
_output_shapes
:*
validate_shape(*
T0
|
Variable_21/Adam_1/readIdentityVariable_21/Adam_1*
_output_shapes
:*
_class
loc:@Variable_21*
T0
�
"Variable_22/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_22*
dtype0*
_output_shapes

:

�
Variable_22/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_22*
dtype0*
	container *
shared_name 
�
Variable_22/Adam/AssignAssignVariable_22/Adam"Variable_22/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_22*
_output_shapes

:
*
validate_shape(*
T0
|
Variable_22/Adam/readIdentityVariable_22/Adam*
_output_shapes

:
*
_class
loc:@Variable_22*
T0
�
$Variable_22/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_22*
dtype0*
_output_shapes

:

�
Variable_22/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_22*
dtype0*
	container *
shared_name 
�
Variable_22/Adam_1/AssignAssignVariable_22/Adam_1$Variable_22/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_22*
_output_shapes

:
*
validate_shape(*
T0
�
Variable_22/Adam_1/readIdentityVariable_22/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_22*
T0
�
"Variable_23/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_23*
dtype0*
_output_shapes
:

�
Variable_23/Adam
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_23*
dtype0*
	container *
shared_name 
�
Variable_23/Adam/AssignAssignVariable_23/Adam"Variable_23/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_23*
_output_shapes
:
*
validate_shape(*
T0
x
Variable_23/Adam/readIdentityVariable_23/Adam*
_output_shapes
:
*
_class
loc:@Variable_23*
T0
�
$Variable_23/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_23*
dtype0*
_output_shapes
:

�
Variable_23/Adam_1
VariableV2*
shape:
*
_output_shapes
:
*
_class
loc:@Variable_23*
dtype0*
	container *
shared_name 
�
Variable_23/Adam_1/AssignAssignVariable_23/Adam_1$Variable_23/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_23*
_output_shapes
:
*
validate_shape(*
T0
|
Variable_23/Adam_1/readIdentityVariable_23/Adam_1*
_output_shapes
:
*
_class
loc:@Variable_23*
T0
�
"Variable_24/Adam/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_24*
dtype0*
_output_shapes

:

�
Variable_24/Adam
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_24*
dtype0*
	container *
shared_name 
�
Variable_24/Adam/AssignAssignVariable_24/Adam"Variable_24/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_24*
_output_shapes

:
*
validate_shape(*
T0
|
Variable_24/Adam/readIdentityVariable_24/Adam*
_output_shapes

:
*
_class
loc:@Variable_24*
T0
�
$Variable_24/Adam_1/Initializer/ConstConst*
valueB
*    *
_class
loc:@Variable_24*
dtype0*
_output_shapes

:

�
Variable_24/Adam_1
VariableV2*
shape
:
*
_output_shapes

:
*
_class
loc:@Variable_24*
dtype0*
	container *
shared_name 
�
Variable_24/Adam_1/AssignAssignVariable_24/Adam_1$Variable_24/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_24*
_output_shapes

:
*
validate_shape(*
T0
�
Variable_24/Adam_1/readIdentityVariable_24/Adam_1*
_output_shapes

:
*
_class
loc:@Variable_24*
T0
�
"Variable_25/Adam/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_25*
dtype0*
_output_shapes
:
�
Variable_25/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_25*
dtype0*
	container *
shared_name 
�
Variable_25/Adam/AssignAssignVariable_25/Adam"Variable_25/Adam/Initializer/Const*
use_locking(*
_class
loc:@Variable_25*
_output_shapes
:*
validate_shape(*
T0
x
Variable_25/Adam/readIdentityVariable_25/Adam*
_output_shapes
:*
_class
loc:@Variable_25*
T0
�
$Variable_25/Adam_1/Initializer/ConstConst*
valueB*    *
_class
loc:@Variable_25*
dtype0*
_output_shapes
:
�
Variable_25/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_25*
dtype0*
	container *
shared_name 
�
Variable_25/Adam_1/AssignAssignVariable_25/Adam_1$Variable_25/Adam_1/Initializer/Const*
use_locking(*
_class
loc:@Variable_25*
_output_shapes
:*
validate_shape(*
T0
|
Variable_25/Adam_1/readIdentityVariable_25/Adam_1*
_output_shapes
:*
_class
loc:@Variable_25*
T0
W
Adam/learning_rateConst*
valueB
 *o�:*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
valueB
 *w�?*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w�+2*
_output_shapes
: *
dtype0
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable*
_output_shapes

:d*
T0
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_1*
_output_shapes
:*
T0
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_2*
_output_shapes

:
*
T0
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_3*
_output_shapes
:
*
T0
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_4*
_output_shapes

:
*
T0
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_5*
_output_shapes
:*
T0
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_6*
_output_shapes

:*
T0
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_3_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_7*
_output_shapes
:*
T0
�
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_4_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_8*
_output_shapes

:*
T0
�
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_4_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_9*
_output_shapes
:*
T0
�
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_5_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_10*
_output_shapes

:
*
T0
�
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_5_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_11*
_output_shapes
:
*
T0
�
!Adam/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_6_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_12*
_output_shapes

:
*
T0
�
!Adam/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_6_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_13*
_output_shapes
:*
T0
�
!Adam/update_Variable_14/ApplyAdam	ApplyAdamVariable_14Variable_14/AdamVariable_14/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_7_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_14*
_output_shapes

:
*
T0
�
!Adam/update_Variable_15/ApplyAdam	ApplyAdamVariable_15Variable_15/AdamVariable_15/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_7_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_15*
_output_shapes
:
*
T0
�
!Adam/update_Variable_16/ApplyAdam	ApplyAdamVariable_16Variable_16/AdamVariable_16/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_8_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_16*
_output_shapes

:
*
T0
�
!Adam/update_Variable_17/ApplyAdam	ApplyAdamVariable_17Variable_17/AdamVariable_17/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_8_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_17*
_output_shapes
:*
T0
�
!Adam/update_Variable_18/ApplyAdam	ApplyAdamVariable_18Variable_18/AdamVariable_18/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_9_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_18*
_output_shapes

:*
T0
�
!Adam/update_Variable_19/ApplyAdam	ApplyAdamVariable_19Variable_19/AdamVariable_19/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_9_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_19*
_output_shapes
:*
T0
�
!Adam/update_Variable_20/ApplyAdam	ApplyAdamVariable_20Variable_20/AdamVariable_20/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/MatMul_10_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_20*
_output_shapes

:*
T0
�
!Adam/update_Variable_21/ApplyAdam	ApplyAdamVariable_21Variable_21/AdamVariable_21/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Add_10_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_21*
_output_shapes
:*
T0
�
!Adam/update_Variable_22/ApplyAdam	ApplyAdamVariable_22Variable_22/AdamVariable_22/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/MatMul_11_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_22*
_output_shapes

:
*
T0
�
!Adam/update_Variable_23/ApplyAdam	ApplyAdamVariable_23Variable_23/AdamVariable_23/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Add_11_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_23*
_output_shapes
:
*
T0
�
!Adam/update_Variable_24/ApplyAdam	ApplyAdamVariable_24Variable_24/AdamVariable_24/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/MatMul_13_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_24*
_output_shapes

:
*
T0
�
!Adam/update_Variable_25/ApplyAdam	ApplyAdamVariable_25Variable_25/AdamVariable_25/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Add_13_grad/tuple/control_dependency_1*
use_locking( *
_class
loc:@Variable_25*
_output_shapes
:*
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam*
_output_shapes
: *
_class
loc:@Variable*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
T0
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam"^Adam/update_Variable_12/ApplyAdam"^Adam/update_Variable_13/ApplyAdam"^Adam/update_Variable_14/ApplyAdam"^Adam/update_Variable_15/ApplyAdam"^Adam/update_Variable_16/ApplyAdam"^Adam/update_Variable_17/ApplyAdam"^Adam/update_Variable_18/ApplyAdam"^Adam/update_Variable_19/ApplyAdam"^Adam/update_Variable_20/ApplyAdam"^Adam/update_Variable_21/ApplyAdam"^Adam/update_Variable_22/ApplyAdam"^Adam/update_Variable_23/ApplyAdam"^Adam/update_Variable_24/ApplyAdam"^Adam/update_Variable_25/ApplyAdam^Adam/Assign^Adam/Assign_1
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_11/Adam/Assign^Variable_11/Adam_1/Assign^Variable_12/Adam/Assign^Variable_12/Adam_1/Assign^Variable_13/Adam/Assign^Variable_13/Adam_1/Assign^Variable_14/Adam/Assign^Variable_14/Adam_1/Assign^Variable_15/Adam/Assign^Variable_15/Adam_1/Assign^Variable_16/Adam/Assign^Variable_16/Adam_1/Assign^Variable_17/Adam/Assign^Variable_17/Adam_1/Assign^Variable_18/Adam/Assign^Variable_18/Adam_1/Assign^Variable_19/Adam/Assign^Variable_19/Adam_1/Assign^Variable_20/Adam/Assign^Variable_20/Adam_1/Assign^Variable_21/Adam/Assign^Variable_21/Adam_1/Assign^Variable_22/Adam/Assign^Variable_22/Adam_1/Assign^Variable_23/Adam/Assign^Variable_23/Adam_1/Assign^Variable_24/Adam/Assign^Variable_24/Adam_1/Assign^Variable_25/Adam/Assign^Variable_25/Adam_1/Assign
R
ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
d
ArgMaxArgMaxAdd_13ArgMax/dimension*#
_output_shapes
:���������*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*#
_output_shapes
:���������*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:���������*
T0	
R
Cast_1CastEqual*#
_output_shapes
:���������*

SrcT0
*

DstT0
Q
Const_1Const*
valueB: *
_output_shapes
:*
dtype0
]
Mean_1MeanCast_1Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0">��p