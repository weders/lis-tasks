       ЃK"	   YО@жAbrain.Event:2 ­wьё       тC	еА(YО@жA"пу
]
PlaceholderPlaceholder*'
_output_shapes
:џџџџџџџџџd*
dtype0*
shape: 
_
Placeholder_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape: 
d
random_normal/shapeConst*
_output_shapes
:*
valueB"d   2   *
dtype0
W
random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
Y
random_normal/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes

:d2
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
_output_shapes

:d2*
T0
d
random_normalAddrandom_normal/mulrandom_normal/mean*
_output_shapes

:d2*
T0
|
Variable
VariableV2*
_output_shapes

:d2*
shared_name *
	container *
shape
:d2*
dtype0
Ё
Variable/AssignAssignVariablerandom_normal*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking(*
_output_shapes

:d2
i
Variable/readIdentityVariable*
_output_shapes

:d2*
T0*
_class
loc:@Variable
_
random_normal_1/shapeConst*
_output_shapes
:*
valueB:2*
dtype0
Y
random_normal_1/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_1/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes
:2
}
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
_output_shapes
:2*
T0
f
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes
:2*
T0
v

Variable_1
VariableV2*
_output_shapes
:2*
shared_name *
	container *
shape:2*
dtype0
Ѕ
Variable_1/AssignAssign
Variable_1random_normal_1*
validate_shape(*
T0*
_class
loc:@Variable_1*
use_locking(*
_output_shapes
:2
k
Variable_1/readIdentity
Variable_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_1
f
random_normal_2/shapeConst*
_output_shapes
:*
valueB"2   2   *
dtype0
Y
random_normal_2/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_2/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ђ
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes

:22

random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes

:22*
T0
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes

:22*
T0
~

Variable_2
VariableV2*
_output_shapes

:22*
shared_name *
	container *
shape
:22*
dtype0
Љ
Variable_2/AssignAssign
Variable_2random_normal_2*
validate_shape(*
T0*
_class
loc:@Variable_2*
use_locking(*
_output_shapes

:22
o
Variable_2/readIdentity
Variable_2*
_output_shapes

:22*
T0*
_class
loc:@Variable_2
_
random_normal_3/shapeConst*
_output_shapes
:*
valueB:2*
dtype0
Y
random_normal_3/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_3/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes
:2
}
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes
:2*
T0
f
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
_output_shapes
:2*
T0
v

Variable_3
VariableV2*
_output_shapes
:2*
shared_name *
	container *
shape:2*
dtype0
Ѕ
Variable_3/AssignAssign
Variable_3random_normal_3*
validate_shape(*
T0*
_class
loc:@Variable_3*
use_locking(*
_output_shapes
:2
k
Variable_3/readIdentity
Variable_3*
_output_shapes
:2*
T0*
_class
loc:@Variable_3
f
random_normal_4/shapeConst*
_output_shapes
:*
valueB"2   d   *
dtype0
Y
random_normal_4/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_4/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ђ
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes

:2d

random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes

:2d*
T0
j
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
_output_shapes

:2d*
T0
~

Variable_4
VariableV2*
_output_shapes

:2d*
shared_name *
	container *
shape
:2d*
dtype0
Љ
Variable_4/AssignAssign
Variable_4random_normal_4*
validate_shape(*
T0*
_class
loc:@Variable_4*
use_locking(*
_output_shapes

:2d
o
Variable_4/readIdentity
Variable_4*
_output_shapes

:2d*
T0*
_class
loc:@Variable_4
_
random_normal_5/shapeConst*
_output_shapes
:*
valueB:d*
dtype0
Y
random_normal_5/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_5/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes
:d
}
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes
:d*
T0
f
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
_output_shapes
:d*
T0
v

Variable_5
VariableV2*
_output_shapes
:d*
shared_name *
	container *
shape:d*
dtype0
Ѕ
Variable_5/AssignAssign
Variable_5random_normal_5*
validate_shape(*
T0*
_class
loc:@Variable_5*
use_locking(*
_output_shapes
:d
k
Variable_5/readIdentity
Variable_5*
_output_shapes
:d*
T0*
_class
loc:@Variable_5
f
random_normal_6/shapeConst*
_output_shapes
:*
valueB"d   2   *
dtype0
Y
random_normal_6/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_6/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ђ
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes

:d2

random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
_output_shapes

:d2*
T0
j
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
_output_shapes

:d2*
T0
~

Variable_6
VariableV2*
_output_shapes

:d2*
shared_name *
	container *
shape
:d2*
dtype0
Љ
Variable_6/AssignAssign
Variable_6random_normal_6*
validate_shape(*
T0*
_class
loc:@Variable_6*
use_locking(*
_output_shapes

:d2
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:d2*
T0*
_class
loc:@Variable_6
_
random_normal_7/shapeConst*
_output_shapes
:*
valueB:2*
dtype0
Y
random_normal_7/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_7/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes
:2
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes
:2*
T0
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes
:2*
T0
v

Variable_7
VariableV2*
_output_shapes
:2*
shared_name *
	container *
shape:2*
dtype0
Ѕ
Variable_7/AssignAssign
Variable_7random_normal_7*
validate_shape(*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes
:2
k
Variable_7/readIdentity
Variable_7*
_output_shapes
:2*
T0*
_class
loc:@Variable_7
f
random_normal_8/shapeConst*
_output_shapes
:*
valueB"2   2   *
dtype0
Y
random_normal_8/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_8/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Ђ
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes

:22

random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
_output_shapes

:22*
T0
j
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes

:22*
T0
~

Variable_8
VariableV2*
_output_shapes

:22*
shared_name *
	container *
shape
:22*
dtype0
Љ
Variable_8/AssignAssign
Variable_8random_normal_8*
validate_shape(*
T0*
_class
loc:@Variable_8*
use_locking(*
_output_shapes

:22
o
Variable_8/readIdentity
Variable_8*
_output_shapes

:22*
T0*
_class
loc:@Variable_8
_
random_normal_9/shapeConst*
_output_shapes
:*
valueB:2*
dtype0
Y
random_normal_9/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
[
random_normal_9/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0

$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes
:2
}
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
_output_shapes
:2*
T0
f
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
_output_shapes
:2*
T0
v

Variable_9
VariableV2*
_output_shapes
:2*
shared_name *
	container *
shape:2*
dtype0
Ѕ
Variable_9/AssignAssign
Variable_9random_normal_9*
validate_shape(*
T0*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:2
k
Variable_9/readIdentity
Variable_9*
_output_shapes
:2*
T0*
_class
loc:@Variable_9
g
random_normal_10/shapeConst*
_output_shapes
:*
valueB"2      *
dtype0
Z
random_normal_10/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
random_normal_10/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Є
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes

:2

random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
_output_shapes

:2*
T0
m
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
_output_shapes

:2*
T0

Variable_10
VariableV2*
_output_shapes

:2*
shared_name *
	container *
shape
:2*
dtype0
­
Variable_10/AssignAssignVariable_10random_normal_10*
validate_shape(*
T0*
_class
loc:@Variable_10*
use_locking(*
_output_shapes

:2
r
Variable_10/readIdentityVariable_10*
_output_shapes

:2*
T0*
_class
loc:@Variable_10
`
random_normal_11/shapeConst*
_output_shapes
:*
valueB:*
dtype0
Z
random_normal_11/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0
\
random_normal_11/stddevConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
 
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*
seed2 *
T0*

seed *
dtype0*
_output_shapes
:

random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
_output_shapes
:*
T0
i
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
_output_shapes
:*
T0
w
Variable_11
VariableV2*
_output_shapes
:*
shared_name *
	container *
shape:*
dtype0
Љ
Variable_11/AssignAssignVariable_11random_normal_11*
validate_shape(*
T0*
_class
loc:@Variable_11*
use_locking(*
_output_shapes
:
n
Variable_11/readIdentityVariable_11*
_output_shapes
:*
T0*
_class
loc:@Variable_11

MatMulMatMulPlaceholderVariable/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b( 
U
AddAddMatMulVariable_1/read*'
_output_shapes
:џџџџџџџџџ2*
T0
C
ReluReluAdd*'
_output_shapes
:џџџџџџџџџ2*
T0

MatMul_1MatMulReluVariable_2/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b( 
Y
Add_1AddMatMul_1Variable_3/read*'
_output_shapes
:џџџџџџџџџ2*
T0
G
Relu_1ReluAdd_1*'
_output_shapes
:џџџџџџџџџ2*
T0

MatMul_2MatMulRelu_1Variable_4/read*'
_output_shapes
:џџџџџџџџџd*
T0*
transpose_a( *
transpose_b( 
Y
Add_2AddMatMul_2Variable_5/read*'
_output_shapes
:џџџџџџџџџd*
T0
G
Relu_2ReluAdd_2*'
_output_shapes
:џџџџџџџџџd*
T0

MatMul_3MatMulRelu_2Variable_6/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b( 
Y
Add_3AddMatMul_3Variable_7/read*'
_output_shapes
:џџџџџџџџџ2*
T0
G
Relu_3ReluAdd_3*'
_output_shapes
:џџџџџџџџџ2*
T0

MatMul_4MatMulRelu_3Variable_8/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b( 
Y
Add_4AddMatMul_4Variable_9/read*'
_output_shapes
:џџџџџџџџџ2*
T0
G
Relu_4ReluAdd_4*'
_output_shapes
:џџџџџџџџџ2*
T0

MatMul_5MatMulRelu_4Variable_10/read*'
_output_shapes
:џџџџџџџџџ*
T0*
transpose_a( *
transpose_b( 
Z
Add_5AddMatMul_5Variable_11/read*'
_output_shapes
:џџџџџџџџџ*
T0
G
Relu_5ReluAdd_5*'
_output_shapes
:џџџџџџџџџ*
T0

MatMul_6MatMulRelu_4Variable_10/read*'
_output_shapes
:џџџџџџџџџ*
T0*
transpose_a( *
transpose_b( 
Z
Add_6AddMatMul_6Variable_11/read*'
_output_shapes
:џџџџџџџџџ*
T0
F
RankConst*
_output_shapes
: *
value	B :*
dtype0
J
ShapeShapeAdd_6*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
_output_shapes
: *
value	B :*
dtype0
L
Shape_1ShapeAdd_6*
out_type0*
T0*
_output_shapes
:
G
Sub/yConst*
_output_shapes
: *
value	B :*
dtype0
:
SubSubRank_1Sub/y*
_output_shapes
: *
T0
R
Slice/beginPackSub*
N*
_output_shapes
:*
T0*

axis 
T

Slice/sizeConst*
_output_shapes
:*
valueB:*
dtype0
b
SliceSliceShape_1Slice/begin
Slice/size*
_output_shapes
:*
T0*
Index0
b
concat/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
q
concatConcatV2concat/values_0Sliceconcat/axis*
N*
_output_shapes
:*
T0*

Tidx0
j
ReshapeReshapeAdd_6concat*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0
H
Rank_2Const*
_output_shapes
: *
value	B :*
dtype0
T
Shape_2ShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
I
Sub_1/yConst*
_output_shapes
: *
value	B :*
dtype0
>
Sub_1SubRank_2Sub_1/y*
_output_shapes
: *
T0
V
Slice_1/beginPackSub_1*
N*
_output_shapes
:*
T0*

axis 
V
Slice_1/sizeConst*
_output_shapes
:*
valueB:*
dtype0
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
_output_shapes
:*
T0*
Index0
d
concat_1/values_0Const*
_output_shapes
:*
valueB:
џџџџџџџџџ*
dtype0
O
concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
v
	Reshape_1ReshapePlaceholder_1concat_1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0*
Tshape0

SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*?
_output_shapes-
+:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ*
T0
I
Sub_2/yConst*
_output_shapes
: *
value	B :*
dtype0
<
Sub_2SubRankSub_2/y*
_output_shapes
: *
T0
W
Slice_2/beginConst*
_output_shapes
:*
valueB: *
dtype0
U
Slice_2/sizePackSub_2*
N*
_output_shapes
:*
T0*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*#
_output_shapes
:џџџџџџџџџ*
T0*
Index0
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
\
MeanMean	Reshape_2Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
R
gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
T
gradients/ConstConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_2*
out_type0*
T0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
Є
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0

;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
_output_shapes
: *
valueB :
џџџџџџџџџ*
dtype0
т
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*
T0*

Tdim0
Ь
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
T0
a
gradients/Reshape_grad/ShapeShapeAdd_6*
out_type0*
T0*
_output_shapes
:
Й
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
b
gradients/Add_6_grad/ShapeShapeMatMul_6*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_6_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
К
*gradients/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_6_grad/Shapegradients/Add_6_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/Add_6_grad/SumSumgradients/Reshape_grad/Reshape*gradients/Add_6_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_6_grad/ReshapeReshapegradients/Add_6_grad/Sumgradients/Add_6_grad/Shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Џ
gradients/Add_6_grad/Sum_1Sumgradients/Reshape_grad/Reshape,gradients/Add_6_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_6_grad/Reshape_1Reshapegradients/Add_6_grad/Sum_1gradients/Add_6_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
m
%gradients/Add_6_grad/tuple/group_depsNoOp^gradients/Add_6_grad/Reshape^gradients/Add_6_grad/Reshape_1
т
-gradients/Add_6_grad/tuple/control_dependencyIdentitygradients/Add_6_grad/Reshape&^gradients/Add_6_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ*
T0*/
_class%
#!loc:@gradients/Add_6_grad/Reshape
л
/gradients/Add_6_grad/tuple/control_dependency_1Identitygradients/Add_6_grad/Reshape_1&^gradients/Add_6_grad/tuple/group_deps*
_output_shapes
:*
T0*1
_class'
%#loc:@gradients/Add_6_grad/Reshape_1
С
gradients/MatMul_6_grad/MatMulMatMul-gradients/Add_6_grad/tuple/control_dependencyVariable_10/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b(
А
 gradients/MatMul_6_grad/MatMul_1MatMulRelu_4-gradients/Add_6_grad/tuple/control_dependency*
_output_shapes

:2*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_6_grad/tuple/group_depsNoOp^gradients/MatMul_6_grad/MatMul!^gradients/MatMul_6_grad/MatMul_1
ь
0gradients/MatMul_6_grad/tuple/control_dependencyIdentitygradients/MatMul_6_grad/MatMul)^gradients/MatMul_6_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*1
_class'
%#loc:@gradients/MatMul_6_grad/MatMul
щ
2gradients/MatMul_6_grad/tuple/control_dependency_1Identity gradients/MatMul_6_grad/MatMul_1)^gradients/MatMul_6_grad/tuple/group_deps*
_output_shapes

:2*
T0*3
_class)
'%loc:@gradients/MatMul_6_grad/MatMul_1

gradients/Relu_4_grad/ReluGradReluGrad0gradients/MatMul_6_grad/tuple/control_dependencyRelu_4*'
_output_shapes
:џџџџџџџџџ2*
T0
b
gradients/Add_4_grad/ShapeShapeMatMul_4*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_4_grad/Shape_1Const*
_output_shapes
:*
valueB:2*
dtype0
К
*gradients/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_4_grad/Shapegradients/Add_4_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/Add_4_grad/SumSumgradients/Relu_4_grad/ReluGrad*gradients/Add_4_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_4_grad/ReshapeReshapegradients/Add_4_grad/Sumgradients/Add_4_grad/Shape*'
_output_shapes
:џџџџџџџџџ2*
T0*
Tshape0
Џ
gradients/Add_4_grad/Sum_1Sumgradients/Relu_4_grad/ReluGrad,gradients/Add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_4_grad/Reshape_1Reshapegradients/Add_4_grad/Sum_1gradients/Add_4_grad/Shape_1*
_output_shapes
:2*
T0*
Tshape0
m
%gradients/Add_4_grad/tuple/group_depsNoOp^gradients/Add_4_grad/Reshape^gradients/Add_4_grad/Reshape_1
т
-gradients/Add_4_grad/tuple/control_dependencyIdentitygradients/Add_4_grad/Reshape&^gradients/Add_4_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*/
_class%
#!loc:@gradients/Add_4_grad/Reshape
л
/gradients/Add_4_grad/tuple/control_dependency_1Identitygradients/Add_4_grad/Reshape_1&^gradients/Add_4_grad/tuple/group_deps*
_output_shapes
:2*
T0*1
_class'
%#loc:@gradients/Add_4_grad/Reshape_1
Р
gradients/MatMul_4_grad/MatMulMatMul-gradients/Add_4_grad/tuple/control_dependencyVariable_8/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b(
А
 gradients/MatMul_4_grad/MatMul_1MatMulRelu_3-gradients/Add_4_grad/tuple/control_dependency*
_output_shapes

:22*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
ь
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul
щ
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*
_output_shapes

:22*
T0*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1

gradients/Relu_3_grad/ReluGradReluGrad0gradients/MatMul_4_grad/tuple/control_dependencyRelu_3*'
_output_shapes
:џџџџџџџџџ2*
T0
b
gradients/Add_3_grad/ShapeShapeMatMul_3*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_3_grad/Shape_1Const*
_output_shapes
:*
valueB:2*
dtype0
К
*gradients/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_3_grad/Shapegradients/Add_3_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/Add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/Add_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_3_grad/ReshapeReshapegradients/Add_3_grad/Sumgradients/Add_3_grad/Shape*'
_output_shapes
:џџџџџџџџџ2*
T0*
Tshape0
Џ
gradients/Add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/Add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_3_grad/Reshape_1Reshapegradients/Add_3_grad/Sum_1gradients/Add_3_grad/Shape_1*
_output_shapes
:2*
T0*
Tshape0
m
%gradients/Add_3_grad/tuple/group_depsNoOp^gradients/Add_3_grad/Reshape^gradients/Add_3_grad/Reshape_1
т
-gradients/Add_3_grad/tuple/control_dependencyIdentitygradients/Add_3_grad/Reshape&^gradients/Add_3_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*/
_class%
#!loc:@gradients/Add_3_grad/Reshape
л
/gradients/Add_3_grad/tuple/control_dependency_1Identitygradients/Add_3_grad/Reshape_1&^gradients/Add_3_grad/tuple/group_deps*
_output_shapes
:2*
T0*1
_class'
%#loc:@gradients/Add_3_grad/Reshape_1
Р
gradients/MatMul_3_grad/MatMulMatMul-gradients/Add_3_grad/tuple/control_dependencyVariable_6/read*'
_output_shapes
:џџџџџџџџџd*
T0*
transpose_a( *
transpose_b(
А
 gradients/MatMul_3_grad/MatMul_1MatMulRelu_2-gradients/Add_3_grad/tuple/control_dependency*
_output_shapes

:d2*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
ь
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul
щ
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
_output_shapes

:d2*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1

gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_3_grad/tuple/control_dependencyRelu_2*'
_output_shapes
:џџџџџџџџџd*
T0
b
gradients/Add_2_grad/ShapeShapeMatMul_2*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_2_grad/Shape_1Const*
_output_shapes
:*
valueB:d*
dtype0
К
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/Add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*'
_output_shapes
:џџџџџџџџџd*
T0*
Tshape0
Џ
gradients/Add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
_output_shapes
:d*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
т
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
л
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
_output_shapes
:d*
T0*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1
Р
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b(
А
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/Add_2_grad/tuple/control_dependency*
_output_shapes

:2d*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
ь
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul
щ
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
_output_shapes

:2d*
T0*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1

gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*'
_output_shapes
:џџџџџџџџџ2*
T0
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
_output_shapes
:*
valueB:2*
dtype0
К
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ћ
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*'
_output_shapes
:џџџџџџџџџ2*
T0*
Tshape0
Џ
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
_output_shapes
:2*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
т
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
л
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
_output_shapes
:2*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1
Р
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_2/read*'
_output_shapes
:џџџџџџџџџ2*
T0*
transpose_a( *
transpose_b(
Ў
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/Add_1_grad/tuple/control_dependency*
_output_shapes

:22*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ь
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
щ
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
_output_shapes

:22*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1

gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*'
_output_shapes
:џџџџџџџџџ2*
T0
^
gradients/Add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
d
gradients/Add_grad/Shape_1Const*
_output_shapes
:*
valueB:2*
dtype0
Д
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ѕ
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*'
_output_shapes
:џџџџџџџџџ2*
T0*
Tshape0
Љ
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
_output_shapes
:2*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
к
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ2*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape
г
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
_output_shapes
:2*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1
К
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable/read*'
_output_shapes
:џџџџџџџџџd*
T0*
transpose_a( *
transpose_b(
Б
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/Add_grad/tuple/control_dependency*
_output_shapes

:d2*
T0*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ф
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџd*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
с
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
_output_shapes

:d2*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
{
beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@Variable*
dtype0

beta1_power
VariableV2*
_class
loc:@Variable*
	container *
shape: *
_output_shapes
: *
shared_name *
dtype0
Ћ
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking(*
_output_shapes
: 
g
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@Variable
{
beta2_power/initial_valueConst*
_output_shapes
: *
valueB
 *wО?*
_class
loc:@Variable*
dtype0

beta2_power
VariableV2*
_class
loc:@Variable*
	container *
shape: *
_output_shapes
: *
shared_name *
dtype0
Ћ
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking(*
_output_shapes
: 
g
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@Variable

Variable/Adam/Initializer/ConstConst*
_output_shapes

:d2*
valueBd2*    *
_class
loc:@Variable*
dtype0

Variable/Adam
VariableV2*
_class
loc:@Variable*
shape
:d2*
_output_shapes

:d2*
shared_name *
	container *
dtype0
Н
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking(*
_output_shapes

:d2
s
Variable/Adam/readIdentityVariable/Adam*
_output_shapes

:d2*
T0*
_class
loc:@Variable

!Variable/Adam_1/Initializer/ConstConst*
_output_shapes

:d2*
valueBd2*    *
_class
loc:@Variable*
dtype0
 
Variable/Adam_1
VariableV2*
_class
loc:@Variable*
shape
:d2*
_output_shapes

:d2*
shared_name *
	container *
dtype0
У
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking(*
_output_shapes

:d2
w
Variable/Adam_1/readIdentityVariable/Adam_1*
_output_shapes

:d2*
T0*
_class
loc:@Variable

!Variable_1/Adam/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_1*
dtype0

Variable_1/Adam
VariableV2*
_class
loc:@Variable_1*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
С
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_1*
use_locking(*
_output_shapes
:2
u
Variable_1/Adam/readIdentityVariable_1/Adam*
_output_shapes
:2*
T0*
_class
loc:@Variable_1

#Variable_1/Adam_1/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_1*
dtype0

Variable_1/Adam_1
VariableV2*
_class
loc:@Variable_1*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
Ч
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_1*
use_locking(*
_output_shapes
:2
y
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_1

!Variable_2/Adam/Initializer/ConstConst*
_output_shapes

:22*
valueB22*    *
_class
loc:@Variable_2*
dtype0
Ђ
Variable_2/Adam
VariableV2*
_class
loc:@Variable_2*
shape
:22*
_output_shapes

:22*
shared_name *
	container *
dtype0
Х
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_2*
use_locking(*
_output_shapes

:22
y
Variable_2/Adam/readIdentityVariable_2/Adam*
_output_shapes

:22*
T0*
_class
loc:@Variable_2

#Variable_2/Adam_1/Initializer/ConstConst*
_output_shapes

:22*
valueB22*    *
_class
loc:@Variable_2*
dtype0
Є
Variable_2/Adam_1
VariableV2*
_class
loc:@Variable_2*
shape
:22*
_output_shapes

:22*
shared_name *
	container *
dtype0
Ы
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_2*
use_locking(*
_output_shapes

:22
}
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_output_shapes

:22*
T0*
_class
loc:@Variable_2

!Variable_3/Adam/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_3*
dtype0

Variable_3/Adam
VariableV2*
_class
loc:@Variable_3*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
С
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_3*
use_locking(*
_output_shapes
:2
u
Variable_3/Adam/readIdentityVariable_3/Adam*
_output_shapes
:2*
T0*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_3*
dtype0

Variable_3/Adam_1
VariableV2*
_class
loc:@Variable_3*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
Ч
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_3*
use_locking(*
_output_shapes
:2
y
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_3

!Variable_4/Adam/Initializer/ConstConst*
_output_shapes

:2d*
valueB2d*    *
_class
loc:@Variable_4*
dtype0
Ђ
Variable_4/Adam
VariableV2*
_class
loc:@Variable_4*
shape
:2d*
_output_shapes

:2d*
shared_name *
	container *
dtype0
Х
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_4*
use_locking(*
_output_shapes

:2d
y
Variable_4/Adam/readIdentityVariable_4/Adam*
_output_shapes

:2d*
T0*
_class
loc:@Variable_4

#Variable_4/Adam_1/Initializer/ConstConst*
_output_shapes

:2d*
valueB2d*    *
_class
loc:@Variable_4*
dtype0
Є
Variable_4/Adam_1
VariableV2*
_class
loc:@Variable_4*
shape
:2d*
_output_shapes

:2d*
shared_name *
	container *
dtype0
Ы
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_4*
use_locking(*
_output_shapes

:2d
}
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_output_shapes

:2d*
T0*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/ConstConst*
_output_shapes
:d*
valueBd*    *
_class
loc:@Variable_5*
dtype0

Variable_5/Adam
VariableV2*
_class
loc:@Variable_5*
shape:d*
_output_shapes
:d*
shared_name *
	container *
dtype0
С
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_5*
use_locking(*
_output_shapes
:d
u
Variable_5/Adam/readIdentityVariable_5/Adam*
_output_shapes
:d*
T0*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/ConstConst*
_output_shapes
:d*
valueBd*    *
_class
loc:@Variable_5*
dtype0

Variable_5/Adam_1
VariableV2*
_class
loc:@Variable_5*
shape:d*
_output_shapes
:d*
shared_name *
	container *
dtype0
Ч
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_5*
use_locking(*
_output_shapes
:d
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_5

!Variable_6/Adam/Initializer/ConstConst*
_output_shapes

:d2*
valueBd2*    *
_class
loc:@Variable_6*
dtype0
Ђ
Variable_6/Adam
VariableV2*
_class
loc:@Variable_6*
shape
:d2*
_output_shapes

:d2*
shared_name *
	container *
dtype0
Х
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_6*
use_locking(*
_output_shapes

:d2
y
Variable_6/Adam/readIdentityVariable_6/Adam*
_output_shapes

:d2*
T0*
_class
loc:@Variable_6

#Variable_6/Adam_1/Initializer/ConstConst*
_output_shapes

:d2*
valueBd2*    *
_class
loc:@Variable_6*
dtype0
Є
Variable_6/Adam_1
VariableV2*
_class
loc:@Variable_6*
shape
:d2*
_output_shapes

:d2*
shared_name *
	container *
dtype0
Ы
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_6*
use_locking(*
_output_shapes

:d2
}
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_output_shapes

:d2*
T0*
_class
loc:@Variable_6

!Variable_7/Adam/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_7*
dtype0

Variable_7/Adam
VariableV2*
_class
loc:@Variable_7*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
С
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes
:2
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_output_shapes
:2*
T0*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_7*
dtype0

Variable_7/Adam_1
VariableV2*
_class
loc:@Variable_7*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
Ч
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_7*
use_locking(*
_output_shapes
:2
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_7

!Variable_8/Adam/Initializer/ConstConst*
_output_shapes

:22*
valueB22*    *
_class
loc:@Variable_8*
dtype0
Ђ
Variable_8/Adam
VariableV2*
_class
loc:@Variable_8*
shape
:22*
_output_shapes

:22*
shared_name *
	container *
dtype0
Х
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_8*
use_locking(*
_output_shapes

:22
y
Variable_8/Adam/readIdentityVariable_8/Adam*
_output_shapes

:22*
T0*
_class
loc:@Variable_8

#Variable_8/Adam_1/Initializer/ConstConst*
_output_shapes

:22*
valueB22*    *
_class
loc:@Variable_8*
dtype0
Є
Variable_8/Adam_1
VariableV2*
_class
loc:@Variable_8*
shape
:22*
_output_shapes

:22*
shared_name *
	container *
dtype0
Ы
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_8*
use_locking(*
_output_shapes

:22
}
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
_output_shapes

:22*
T0*
_class
loc:@Variable_8

!Variable_9/Adam/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_9*
dtype0

Variable_9/Adam
VariableV2*
_class
loc:@Variable_9*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
С
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:2
u
Variable_9/Adam/readIdentityVariable_9/Adam*
_output_shapes
:2*
T0*
_class
loc:@Variable_9

#Variable_9/Adam_1/Initializer/ConstConst*
_output_shapes
:2*
valueB2*    *
_class
loc:@Variable_9*
dtype0

Variable_9/Adam_1
VariableV2*
_class
loc:@Variable_9*
shape:2*
_output_shapes
:2*
shared_name *
	container *
dtype0
Ч
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_9*
use_locking(*
_output_shapes
:2
y
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_9

"Variable_10/Adam/Initializer/ConstConst*
_output_shapes

:2*
valueB2*    *
_class
loc:@Variable_10*
dtype0
Є
Variable_10/Adam
VariableV2*
_class
loc:@Variable_10*
shape
:2*
_output_shapes

:2*
shared_name *
	container *
dtype0
Щ
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_10*
use_locking(*
_output_shapes

:2
|
Variable_10/Adam/readIdentityVariable_10/Adam*
_output_shapes

:2*
T0*
_class
loc:@Variable_10

$Variable_10/Adam_1/Initializer/ConstConst*
_output_shapes

:2*
valueB2*    *
_class
loc:@Variable_10*
dtype0
І
Variable_10/Adam_1
VariableV2*
_class
loc:@Variable_10*
shape
:2*
_output_shapes

:2*
shared_name *
	container *
dtype0
Я
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_10*
use_locking(*
_output_shapes

:2

Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
_output_shapes

:2*
T0*
_class
loc:@Variable_10

"Variable_11/Adam/Initializer/ConstConst*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_11*
dtype0

Variable_11/Adam
VariableV2*
_class
loc:@Variable_11*
shape:*
_output_shapes
:*
shared_name *
	container *
dtype0
Х
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_11*
use_locking(*
_output_shapes
:
x
Variable_11/Adam/readIdentityVariable_11/Adam*
_output_shapes
:*
T0*
_class
loc:@Variable_11

$Variable_11/Adam_1/Initializer/ConstConst*
_output_shapes
:*
valueB*    *
_class
loc:@Variable_11*
dtype0

Variable_11/Adam_1
VariableV2*
_class
loc:@Variable_11*
shape:*
_output_shapes
:*
shared_name *
	container *
dtype0
Ы
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/Const*
validate_shape(*
T0*
_class
loc:@Variable_11*
use_locking(*
_output_shapes
:
|
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
_output_shapes
:*
T0*
_class
loc:@Variable_11
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *o:*
dtype0
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
_output_shapes
: *
valueB
 *wО?*
dtype0
Q
Adam/epsilonConst*
_output_shapes
: *
valueB
 *wЬ+2*
dtype0
О
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:d2*
T0*
_class
loc:@Variable*
use_locking( 
С
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_1*
use_locking( 
Ъ
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_output_shapes

:22*
T0*
_class
loc:@Variable_2*
use_locking( 
У
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_3*
use_locking( 
Ъ
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
_output_shapes

:2d*
T0*
_class
loc:@Variable_4*
use_locking( 
У
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
_output_shapes
:d*
T0*
_class
loc:@Variable_5*
use_locking( 
Ъ
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
_output_shapes

:d2*
T0*
_class
loc:@Variable_6*
use_locking( 
У
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_3_grad/tuple/control_dependency_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_7*
use_locking( 
Ъ
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_4_grad/tuple/control_dependency_1*
_output_shapes

:22*
T0*
_class
loc:@Variable_8*
use_locking( 
У
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_4_grad/tuple/control_dependency_1*
_output_shapes
:2*
T0*
_class
loc:@Variable_9*
use_locking( 
Я
!Adam/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_6_grad/tuple/control_dependency_1*
_output_shapes

:2*
T0*
_class
loc:@Variable_10*
use_locking( 
Ш
!Adam/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_6_grad/tuple/control_dependency_1*
_output_shapes
:*
T0*
_class
loc:@Variable_11*
use_locking( 

Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking( *
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
T0*
_class
loc:@Variable*
use_locking( *
_output_shapes
: 
Ю
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam"^Adam/update_Variable_10/ApplyAdam"^Adam/update_Variable_11/ApplyAdam^Adam/Assign^Adam/Assign_1

initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_11/Adam/Assign^Variable_11/Adam_1/Assign
R
ArgMax/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
c
ArgMaxArgMaxAdd_6ArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
value	B :*
dtype0
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*
T0*

Tidx0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

Q
Const_1Const*
_output_shapes
:*
valueB: *
dtype0
]
Mean_1MeanCast_1Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: "+_Ї