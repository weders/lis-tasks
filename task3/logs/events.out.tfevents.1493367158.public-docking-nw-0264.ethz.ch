       ЃK"	  ]О@жAbrain.Event:2P5гьё       тC	kЉ]О@жA"пу
]
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:џџџџџџџџџd*
shape: 
_
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:џџџџџџџџџ*
shape: 
d
random_normal/shapeConst*
dtype0*
valueB"d   2   *
_output_shapes
:
W
random_normal/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Y
random_normal/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*

seed *
_output_shapes

:d2*
seed2 *
T0
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes

:d2
d
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes

:d2
|
Variable
VariableV2*
dtype0*
shared_name *
_output_shapes

:d2*
	container *
shape
:d2
Ё
Variable/AssignAssignVariablerandom_normal*
T0*
_class
loc:@Variable*
_output_shapes

:d2*
validate_shape(*
use_locking(
i
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes

:d2
_
random_normal_1/shapeConst*
dtype0*
valueB:2*
_output_shapes
:
Y
random_normal_1/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_1/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*

seed *
_output_shapes
:2*
seed2 *
T0
}
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes
:2
f
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes
:2
v

Variable_1
VariableV2*
dtype0*
shared_name *
_output_shapes
:2*
	container *
shape:2
Ѕ
Variable_1/AssignAssign
Variable_1random_normal_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:2*
validate_shape(*
use_locking(
k
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:2
f
random_normal_2/shapeConst*
dtype0*
valueB"2   d   *
_output_shapes
:
Y
random_normal_2/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_2/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Ђ
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*

seed *
_output_shapes

:2d*
seed2 *
T0

random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0*
_output_shapes

:2d
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0*
_output_shapes

:2d
~

Variable_2
VariableV2*
dtype0*
shared_name *
_output_shapes

:2d*
	container *
shape
:2d
Љ
Variable_2/AssignAssign
Variable_2random_normal_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d*
validate_shape(*
use_locking(
o
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d
_
random_normal_3/shapeConst*
dtype0*
valueB:d*
_output_shapes
:
Y
random_normal_3/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_3/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
dtype0*

seed *
_output_shapes
:d*
seed2 *
T0
}
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0*
_output_shapes
:d
f
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0*
_output_shapes
:d
v

Variable_3
VariableV2*
dtype0*
shared_name *
_output_shapes
:d*
	container *
shape:d
Ѕ
Variable_3/AssignAssign
Variable_3random_normal_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
validate_shape(*
use_locking(
k
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3*
_output_shapes
:d
f
random_normal_4/shapeConst*
dtype0*
valueB"d   d   *
_output_shapes
:
Y
random_normal_4/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_4/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Ђ
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
dtype0*

seed *
_output_shapes

:dd*
seed2 *
T0

random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0*
_output_shapes

:dd
j
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0*
_output_shapes

:dd
~

Variable_4
VariableV2*
dtype0*
shared_name *
_output_shapes

:dd*
	container *
shape
:dd
Љ
Variable_4/AssignAssign
Variable_4random_normal_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd*
validate_shape(*
use_locking(
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd
_
random_normal_5/shapeConst*
dtype0*
valueB:d*
_output_shapes
:
Y
random_normal_5/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_5/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
dtype0*

seed *
_output_shapes
:d*
seed2 *
T0
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0*
_output_shapes
:d
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes
:d
v

Variable_5
VariableV2*
dtype0*
shared_name *
_output_shapes
:d*
	container *
shape:d
Ѕ
Variable_5/AssignAssign
Variable_5random_normal_5*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
validate_shape(*
use_locking(
k
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5*
_output_shapes
:d
f
random_normal_6/shapeConst*
dtype0*
valueB"d   d   *
_output_shapes
:
Y
random_normal_6/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_6/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Ђ
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*

seed *
_output_shapes

:dd*
seed2 *
T0

random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes

:dd
j
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0*
_output_shapes

:dd
~

Variable_6
VariableV2*
dtype0*
shared_name *
_output_shapes

:dd*
	container *
shape
:dd
Љ
Variable_6/AssignAssign
Variable_6random_normal_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd*
validate_shape(*
use_locking(
o
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd
_
random_normal_7/shapeConst*
dtype0*
valueB:d*
_output_shapes
:
Y
random_normal_7/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_7/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
dtype0*

seed *
_output_shapes
:d*
seed2 *
T0
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0*
_output_shapes
:d
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0*
_output_shapes
:d
v

Variable_7
VariableV2*
dtype0*
shared_name *
_output_shapes
:d*
	container *
shape:d
Ѕ
Variable_7/AssignAssign
Variable_7random_normal_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:d*
validate_shape(*
use_locking(
k
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7*
_output_shapes
:d
f
random_normal_8/shapeConst*
dtype0*
valueB"d   2   *
_output_shapes
:
Y
random_normal_8/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_8/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Ђ
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
dtype0*

seed *
_output_shapes

:d2*
seed2 *
T0

random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
T0*
_output_shapes

:d2
j
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
T0*
_output_shapes

:d2
~

Variable_8
VariableV2*
dtype0*
shared_name *
_output_shapes

:d2*
	container *
shape
:d2
Љ
Variable_8/AssignAssign
Variable_8random_normal_8*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2*
validate_shape(*
use_locking(
o
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2
_
random_normal_9/shapeConst*
dtype0*
valueB:2*
_output_shapes
:
Y
random_normal_9/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
[
random_normal_9/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
dtype0*

seed *
_output_shapes
:2*
seed2 *
T0
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
T0*
_output_shapes
:2
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0*
_output_shapes
:2
v

Variable_9
VariableV2*
dtype0*
shared_name *
_output_shapes
:2*
	container *
shape:2
Ѕ
Variable_9/AssignAssign
Variable_9random_normal_9*
T0*
_class
loc:@Variable_9*
_output_shapes
:2*
validate_shape(*
use_locking(
k
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*
_output_shapes
:2
g
random_normal_10/shapeConst*
dtype0*
valueB"2      *
_output_shapes
:
Z
random_normal_10/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
random_normal_10/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Є
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*
dtype0*

seed *
_output_shapes

:2*
seed2 *
T0

random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
T0*
_output_shapes

:2
m
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
T0*
_output_shapes

:2

Variable_10
VariableV2*
dtype0*
shared_name *
_output_shapes

:2*
	container *
shape
:2
­
Variable_10/AssignAssignVariable_10random_normal_10*
T0*
_class
loc:@Variable_10*
_output_shapes

:2*
validate_shape(*
use_locking(
r
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*
_output_shapes

:2
`
random_normal_11/shapeConst*
dtype0*
valueB:*
_output_shapes
:
Z
random_normal_11/meanConst*
dtype0*
valueB
 *    *
_output_shapes
: 
\
random_normal_11/stddevConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
 
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*
dtype0*

seed *
_output_shapes
:*
seed2 *
T0

random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
T0*
_output_shapes
:
i
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
T0*
_output_shapes
:
w
Variable_11
VariableV2*
dtype0*
shared_name *
_output_shapes
:*
	container *
shape:
Љ
Variable_11/AssignAssignVariable_11random_normal_11*
T0*
_class
loc:@Variable_11*
_output_shapes
:*
validate_shape(*
use_locking(
n
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11*
_output_shapes
:

MatMulMatMulPlaceholderVariable/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ2*
transpose_b( 
U
AddAddMatMulVariable_1/read*
T0*'
_output_shapes
:џџџџџџџџџ2
C
ReluReluAdd*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_1MatMulReluVariable_2/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b( 
Y
Add_1AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:џџџџџџџџџd
G
Relu_1ReluAdd_1*
T0*'
_output_shapes
:џџџџџџџџџd

MatMul_2MatMulRelu_1Variable_4/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b( 
Y
Add_2AddMatMul_2Variable_5/read*
T0*'
_output_shapes
:џџџџџџџџџd
G
Relu_2ReluAdd_2*
T0*'
_output_shapes
:џџџџџџџџџd

MatMul_3MatMulRelu_2Variable_6/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b( 
Y
Add_3AddMatMul_3Variable_7/read*
T0*'
_output_shapes
:џџџџџџџџџd
G
Relu_3ReluAdd_3*
T0*'
_output_shapes
:џџџџџџџџџd

MatMul_4MatMulRelu_3Variable_8/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ2*
transpose_b( 
Y
Add_4AddMatMul_4Variable_9/read*
T0*'
_output_shapes
:џџџџџџџџџ2
G
Relu_4ReluAdd_4*
T0*'
_output_shapes
:џџџџџџџџџ2

MatMul_5MatMulRelu_4Variable_10/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Z
Add_5AddMatMul_5Variable_11/read*
T0*'
_output_shapes
:џџџџџџџџџ
G
Relu_5ReluAdd_5*
T0*'
_output_shapes
:џџџџџџџџџ

MatMul_6MatMulRelu_4Variable_10/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_b( 
Z
Add_6AddMatMul_6Variable_11/read*
T0*'
_output_shapes
:џџџџџџџџџ
F
RankConst*
dtype0*
value	B :*
_output_shapes
: 
J
ShapeShapeAdd_6*
T0*
out_type0*
_output_shapes
:
H
Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
L
Shape_1ShapeAdd_6*
T0*
out_type0*
_output_shapes
:
G
Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
T0*
_output_shapes
:*
N*

axis 
T

Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
_output_shapes
:*
Index0
b
concat/values_0Const*
dtype0*
valueB:
џџџџџџџџџ*
_output_shapes
:
M
concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
q
concatConcatV2concat/values_0Sliceconcat/axis*
T0*

Tidx0*
_output_shapes
:*
N
j
ReshapeReshapeAdd_6concat*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
H
Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
T
Shape_2ShapePlaceholder_1*
T0*
out_type0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
T0*
_output_shapes
:*
N*

axis 
V
Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
_output_shapes
:*
Index0
d
concat_1/values_0Const*
dtype0*
valueB:
џџџџџџџџџ*
_output_shapes
:
O
concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N
v
	Reshape_1ReshapePlaceholder_1concat_1*
T0*
Tshape0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*
T0*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ
I
Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
dtype0*
valueB: *
_output_shapes
:
U
Slice_2/sizePackSub_2*
T0*
_output_shapes
:*
N*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*#
_output_shapes
:џџџџџџџџџ*
Index0
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
\
MeanMean	Reshape_2Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
T
gradients/ConstConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
b
gradients/Mean_grad/ShapeShape	Reshape_2*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
Є
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:џџџџџџџџџ
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
џџџџџџџџџ*
_output_shapes
: 
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:џџџџџџџџџ
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
a
gradients/Reshape_grad/ShapeShapeAdd_6*
T0*
out_type0*
_output_shapes
:
Й
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
b
gradients/Add_6_grad/ShapeShapeMatMul_6*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_6_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
К
*gradients/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_6_grad/Shapegradients/Add_6_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/Add_6_grad/SumSumgradients/Reshape_grad/Reshape*gradients/Add_6_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_6_grad/ReshapeReshapegradients/Add_6_grad/Sumgradients/Add_6_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Џ
gradients/Add_6_grad/Sum_1Sumgradients/Reshape_grad/Reshape,gradients/Add_6_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_6_grad/Reshape_1Reshapegradients/Add_6_grad/Sum_1gradients/Add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/Add_6_grad/tuple/group_depsNoOp^gradients/Add_6_grad/Reshape^gradients/Add_6_grad/Reshape_1
т
-gradients/Add_6_grad/tuple/control_dependencyIdentitygradients/Add_6_grad/Reshape&^gradients/Add_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_6_grad/Reshape*'
_output_shapes
:џџџџџџџџџ
л
/gradients/Add_6_grad/tuple/control_dependency_1Identitygradients/Add_6_grad/Reshape_1&^gradients/Add_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_6_grad/Reshape_1*
_output_shapes
:
С
gradients/MatMul_6_grad/MatMulMatMul-gradients/Add_6_grad/tuple/control_dependencyVariable_10/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ2*
transpose_b(
А
 gradients/MatMul_6_grad/MatMul_1MatMulRelu_4-gradients/Add_6_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:2*
transpose_b( 
t
(gradients/MatMul_6_grad/tuple/group_depsNoOp^gradients/MatMul_6_grad/MatMul!^gradients/MatMul_6_grad/MatMul_1
ь
0gradients/MatMul_6_grad/tuple/control_dependencyIdentitygradients/MatMul_6_grad/MatMul)^gradients/MatMul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_6_grad/MatMul*'
_output_shapes
:џџџџџџџџџ2
щ
2gradients/MatMul_6_grad/tuple/control_dependency_1Identity gradients/MatMul_6_grad/MatMul_1)^gradients/MatMul_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_6_grad/MatMul_1*
_output_shapes

:2

gradients/Relu_4_grad/ReluGradReluGrad0gradients/MatMul_6_grad/tuple/control_dependencyRelu_4*
T0*'
_output_shapes
:џџџџџџџџџ2
b
gradients/Add_4_grad/ShapeShapeMatMul_4*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_4_grad/Shape_1Const*
dtype0*
valueB:2*
_output_shapes
:
К
*gradients/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_4_grad/Shapegradients/Add_4_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/Add_4_grad/SumSumgradients/Relu_4_grad/ReluGrad*gradients/Add_4_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_4_grad/ReshapeReshapegradients/Add_4_grad/Sumgradients/Add_4_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ2
Џ
gradients/Add_4_grad/Sum_1Sumgradients/Relu_4_grad/ReluGrad,gradients/Add_4_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_4_grad/Reshape_1Reshapegradients/Add_4_grad/Sum_1gradients/Add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:2
m
%gradients/Add_4_grad/tuple/group_depsNoOp^gradients/Add_4_grad/Reshape^gradients/Add_4_grad/Reshape_1
т
-gradients/Add_4_grad/tuple/control_dependencyIdentitygradients/Add_4_grad/Reshape&^gradients/Add_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_4_grad/Reshape*'
_output_shapes
:џџџџџџџџџ2
л
/gradients/Add_4_grad/tuple/control_dependency_1Identitygradients/Add_4_grad/Reshape_1&^gradients/Add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_4_grad/Reshape_1*
_output_shapes
:2
Р
gradients/MatMul_4_grad/MatMulMatMul-gradients/Add_4_grad/tuple/control_dependencyVariable_8/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b(
А
 gradients/MatMul_4_grad/MatMul_1MatMulRelu_3-gradients/Add_4_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:d2*
transpose_b( 
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
ь
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul*'
_output_shapes
:џџџџџџџџџd
щ
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1*
_output_shapes

:d2

gradients/Relu_3_grad/ReluGradReluGrad0gradients/MatMul_4_grad/tuple/control_dependencyRelu_3*
T0*'
_output_shapes
:џџџџџџџџџd
b
gradients/Add_3_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_3_grad/Shape_1Const*
dtype0*
valueB:d*
_output_shapes
:
К
*gradients/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_3_grad/Shapegradients/Add_3_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/Add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/Add_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_3_grad/ReshapeReshapegradients/Add_3_grad/Sumgradients/Add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd
Џ
gradients/Add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/Add_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_3_grad/Reshape_1Reshapegradients/Add_3_grad/Sum_1gradients/Add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
m
%gradients/Add_3_grad/tuple/group_depsNoOp^gradients/Add_3_grad/Reshape^gradients/Add_3_grad/Reshape_1
т
-gradients/Add_3_grad/tuple/control_dependencyIdentitygradients/Add_3_grad/Reshape&^gradients/Add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_3_grad/Reshape*'
_output_shapes
:џџџџџџџџџd
л
/gradients/Add_3_grad/tuple/control_dependency_1Identitygradients/Add_3_grad/Reshape_1&^gradients/Add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_3_grad/Reshape_1*
_output_shapes
:d
Р
gradients/MatMul_3_grad/MatMulMatMul-gradients/Add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b(
А
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/Add_3_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:dd*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
ь
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul*'
_output_shapes
:џџџџџџџџџd
щ
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:dd

gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*
T0*'
_output_shapes
:џџџџџџџџџd
b
gradients/Add_2_grad/ShapeShapeMatMul_2*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_2_grad/Shape_1Const*
dtype0*
valueB:d*
_output_shapes
:
К
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/Add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd
Џ
gradients/Add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
т
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape*'
_output_shapes
:џџџџџџџџџd
л
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
_output_shapes
:d
Р
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b(
А
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/Add_2_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:dd*
transpose_b( 
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
ь
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*'
_output_shapes
:џџџџџџџџџd
щ
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
_output_shapes

:dd

gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0*'
_output_shapes
:џџџџџџџџџd
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
dtype0*
valueB:d*
_output_shapes
:
К
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ћ
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd
Џ
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
т
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*'
_output_shapes
:џџџџџџџџџd
л
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
_output_shapes
:d
Р
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ2*
transpose_b(
Ў
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/Add_1_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:2d*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ь
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*'
_output_shapes
:џџџџџџџџџ2
щ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
_output_shapes

:2d

gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:џџџџџџџџџ2
^
gradients/Add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/Add_grad/Shape_1Const*
dtype0*
valueB:2*
_output_shapes
:
Д
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѕ
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ2
Љ
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:2
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
к
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ2
г
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
_output_shapes
:2
К
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable/read*
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_b(
Б
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/Add_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes

:d2*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ф
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd
с
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes

:d2
{
beta1_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *fff?*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
shape: *
dtype0*
_output_shapes
: *
	container 
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
use_locking(
g
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *wО?*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *
_class
loc:@Variable*
shape: *
dtype0*
_output_shapes
: *
	container 
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
use_locking(
g
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable*
_output_shapes
: 

Variable/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable*
valueBd2*    *
_output_shapes

:d2

Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
shape
:d2*
dtype0*
_output_shapes

:d2*
	container 
Н
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/Const*
T0*
_class
loc:@Variable*
_output_shapes

:d2*
validate_shape(*
use_locking(
s
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable*
_output_shapes

:d2

!Variable/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable*
valueBd2*    *
_output_shapes

:d2
 
Variable/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable*
shape
:d2*
dtype0*
_output_shapes

:d2*
	container 
У
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable*
_output_shapes

:d2*
validate_shape(*
use_locking(
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable*
_output_shapes

:d2

!Variable_1/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_1*
valueB2*    *
_output_shapes
:2

Variable_1/Adam
VariableV2*
shared_name *
_class
loc:@Variable_1*
shape:2*
dtype0*
_output_shapes
:2*
	container 
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/Const*
T0*
_class
loc:@Variable_1*
_output_shapes
:2*
validate_shape(*
use_locking(
u
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1*
_output_shapes
:2

#Variable_1/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_1*
valueB2*    *
_output_shapes
:2

Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
shape:2*
dtype0*
_output_shapes
:2*
	container 
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_1*
_output_shapes
:2*
validate_shape(*
use_locking(
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:2

!Variable_2/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_2*
valueB2d*    *
_output_shapes

:2d
Ђ
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
shape
:2d*
dtype0*
_output_shapes

:2d*
	container 
Х
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/Const*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d*
validate_shape(*
use_locking(
y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d

#Variable_2/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_2*
valueB2d*    *
_output_shapes

:2d
Є
Variable_2/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
shape
:2d*
dtype0*
_output_shapes

:2d*
	container 
Ы
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d*
validate_shape(*
use_locking(
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d

!Variable_3/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_3*
valueBd*    *
_output_shapes
:d

Variable_3/Adam
VariableV2*
shared_name *
_class
loc:@Variable_3*
shape:d*
dtype0*
_output_shapes
:d*
	container 
С
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/Const*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
validate_shape(*
use_locking(
u
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3*
_output_shapes
:d

#Variable_3/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_3*
valueBd*    *
_output_shapes
:d

Variable_3/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_3*
shape:d*
dtype0*
_output_shapes
:d*
	container 
Ч
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
validate_shape(*
use_locking(
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:d

!Variable_4/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_4*
valueBdd*    *
_output_shapes

:dd
Ђ
Variable_4/Adam
VariableV2*
shared_name *
_class
loc:@Variable_4*
shape
:dd*
dtype0*
_output_shapes

:dd*
	container 
Х
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/Const*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd*
validate_shape(*
use_locking(
y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd

#Variable_4/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_4*
valueBdd*    *
_output_shapes

:dd
Є
Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
shape
:dd*
dtype0*
_output_shapes

:dd*
	container 
Ы
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd*
validate_shape(*
use_locking(
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd

!Variable_5/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_5*
valueBd*    *
_output_shapes
:d

Variable_5/Adam
VariableV2*
shared_name *
_class
loc:@Variable_5*
shape:d*
dtype0*
_output_shapes
:d*
	container 
С
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/Const*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
validate_shape(*
use_locking(
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5*
_output_shapes
:d

#Variable_5/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_5*
valueBd*    *
_output_shapes
:d

Variable_5/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
shape:d*
dtype0*
_output_shapes
:d*
	container 
Ч
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
validate_shape(*
use_locking(
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:d

!Variable_6/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_6*
valueBdd*    *
_output_shapes

:dd
Ђ
Variable_6/Adam
VariableV2*
shared_name *
_class
loc:@Variable_6*
shape
:dd*
dtype0*
_output_shapes

:dd*
	container 
Х
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/Const*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd*
validate_shape(*
use_locking(
y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd

#Variable_6/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_6*
valueBdd*    *
_output_shapes

:dd
Є
Variable_6/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_6*
shape
:dd*
dtype0*
_output_shapes

:dd*
	container 
Ы
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd*
validate_shape(*
use_locking(
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd

!Variable_7/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_7*
valueBd*    *
_output_shapes
:d

Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
shape:d*
dtype0*
_output_shapes
:d*
	container 
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/Const*
T0*
_class
loc:@Variable_7*
_output_shapes
:d*
validate_shape(*
use_locking(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7*
_output_shapes
:d

#Variable_7/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_7*
valueBd*    *
_output_shapes
:d

Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
shape:d*
dtype0*
_output_shapes
:d*
	container 
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_7*
_output_shapes
:d*
validate_shape(*
use_locking(
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7*
_output_shapes
:d

!Variable_8/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_8*
valueBd2*    *
_output_shapes

:d2
Ђ
Variable_8/Adam
VariableV2*
shared_name *
_class
loc:@Variable_8*
shape
:d2*
dtype0*
_output_shapes

:d2*
	container 
Х
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/Const*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2*
validate_shape(*
use_locking(
y
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2

#Variable_8/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_8*
valueBd2*    *
_output_shapes

:d2
Є
Variable_8/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_8*
shape
:d2*
dtype0*
_output_shapes

:d2*
	container 
Ы
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2*
validate_shape(*
use_locking(
}
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2

!Variable_9/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_9*
valueB2*    *
_output_shapes
:2

Variable_9/Adam
VariableV2*
shared_name *
_class
loc:@Variable_9*
shape:2*
dtype0*
_output_shapes
:2*
	container 
С
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/Const*
T0*
_class
loc:@Variable_9*
_output_shapes
:2*
validate_shape(*
use_locking(
u
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9*
_output_shapes
:2

#Variable_9/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_9*
valueB2*    *
_output_shapes
:2

Variable_9/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_9*
shape:2*
dtype0*
_output_shapes
:2*
	container 
Ч
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_9*
_output_shapes
:2*
validate_shape(*
use_locking(
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9*
_output_shapes
:2

"Variable_10/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_10*
valueB2*    *
_output_shapes

:2
Є
Variable_10/Adam
VariableV2*
shared_name *
_class
loc:@Variable_10*
shape
:2*
dtype0*
_output_shapes

:2*
	container 
Щ
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/Const*
T0*
_class
loc:@Variable_10*
_output_shapes

:2*
validate_shape(*
use_locking(
|
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10*
_output_shapes

:2

$Variable_10/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_10*
valueB2*    *
_output_shapes

:2
І
Variable_10/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_10*
shape
:2*
dtype0*
_output_shapes

:2*
	container 
Я
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_10*
_output_shapes

:2*
validate_shape(*
use_locking(

Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10*
_output_shapes

:2

"Variable_11/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_11*
valueB*    *
_output_shapes
:

Variable_11/Adam
VariableV2*
shared_name *
_class
loc:@Variable_11*
shape:*
dtype0*
_output_shapes
:*
	container 
Х
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/Const*
T0*
_class
loc:@Variable_11*
_output_shapes
:*
validate_shape(*
use_locking(
x
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11*
_output_shapes
:

$Variable_11/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_11*
valueB*    *
_output_shapes
:

Variable_11/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_11*
shape:*
dtype0*
_output_shapes
:*
	container 
Ы
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/Const*
T0*
_class
loc:@Variable_11*
_output_shapes
:*
validate_shape(*
use_locking(
|
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_class
loc:@Variable_11*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
valueB
 *wО?*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
О
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable*
_output_shapes

:d2*
use_locking( 
С
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_1*
_output_shapes
:2*
use_locking( 
Ъ
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_2*
_output_shapes

:2d*
use_locking( 
У
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_3*
_output_shapes
:d*
use_locking( 
Ъ
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_4*
_output_shapes

:dd*
use_locking( 
У
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_5*
_output_shapes
:d*
use_locking( 
Ъ
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_6*
_output_shapes

:dd*
use_locking( 
У
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_7*
_output_shapes
:d*
use_locking( 
Ъ
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_4_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_8*
_output_shapes

:d2*
use_locking( 
У
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_4_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_9*
_output_shapes
:2*
use_locking( 
Я
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_6_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_10*
_output_shapes

:2*
use_locking( 
Ш
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_6_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_11*
_output_shapes
:*
use_locking( 

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
use_locking( 


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
T0*
_class
loc:@Variable*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@Variable*
_output_shapes
: *
validate_shape(*
use_locking( 
Ю
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam^Adam/Assign^Adam/Assign_1

initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_11/Adam/Assign^Variable_11/Adam_1/Assign
R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
c
ArgMaxArgMaxAdd_6ArgMax/dimension*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
T
ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*

Tidx0*#
_output_shapes
:џџџџџџџџџ
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
R
Cast_1CastEqual*

DstT0*#
_output_shapes
:џџџџџџџџџ*

SrcT0

Q
Const_1Const*
dtype0*
valueB: *
_output_shapes
:
]
Mean_1MeanCast_1Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( "ђ0Я