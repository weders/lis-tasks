       �K"	  @ā@�Abrain.Event:2o�0��      F�My	�Kā@�A"��
]
PlaceholderPlaceholder*
dtype0*
shape: *'
_output_shapes
:���������d
_
Placeholder_1Placeholder*
dtype0*
shape: *'
_output_shapes
:���������
d
random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"d   �  
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
dtype0*
T0*
seed2 *

seed *
_output_shapes
:	d�
|
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:	d�
e
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes
:	d�
~
Variable
VariableV2*
dtype0*
shape:	d�*
_output_shapes
:	d�*
	container *
shared_name 
�
Variable/AssignAssignVariablerandom_normal*
_class
loc:@Variable*
T0*
_output_shapes
:	d�*
use_locking(*
validate_shape(
j
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
:	d�
`
random_normal_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
Y
random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*
T0*
seed2 *

seed *
_output_shapes	
:�
~
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes	
:�
g
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes	
:�
x

Variable_1
VariableV2*
dtype0*
shape:�*
_output_shapes	
:�*
	container *
shared_name 
�
Variable_1/AssignAssign
Variable_1random_normal_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
l
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
f
random_normal_2/shapeConst*
dtype0*
_output_shapes
:*
valueB"�  �  
Y
random_normal_2/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_2/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*
T0*
seed2 *

seed * 
_output_shapes
:
��
�
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
T0* 
_output_shapes
:
��
l
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
T0* 
_output_shapes
:
��
�

Variable_2
VariableV2*
dtype0*
shape:
��* 
_output_shapes
:
��*
	container *
shared_name 
�
Variable_2/AssignAssign
Variable_2random_normal_2*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��*
use_locking(*
validate_shape(
q
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��
`
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
Y
random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
dtype0*
T0*
seed2 *

seed *
_output_shapes	
:�
~
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
T0*
_output_shapes	
:�
g
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0*
_output_shapes	
:�
x

Variable_3
VariableV2*
dtype0*
shape:�*
_output_shapes	
:�*
	container *
shared_name 
�
Variable_3/AssignAssign
Variable_3random_normal_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
l
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
f
random_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"�  �  
Y
random_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_4/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
dtype0*
T0*
seed2 *

seed * 
_output_shapes
:
��
�
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
T0* 
_output_shapes
:
��
l
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0* 
_output_shapes
:
��
�

Variable_4
VariableV2*
dtype0*
shape:
��* 
_output_shapes
:
��*
	container *
shared_name 
�
Variable_4/AssignAssign
Variable_4random_normal_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��*
use_locking(*
validate_shape(
q
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
`
random_normal_5/shapeConst*
dtype0*
_output_shapes
:*
valueB:�
Y
random_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*
dtype0*
T0*
seed2 *

seed *
_output_shapes	
:�
~
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0*
_output_shapes	
:�
g
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes	
:�
x

Variable_5
VariableV2*
dtype0*
shape:�*
_output_shapes	
:�*
	container *
shared_name 
�
Variable_5/AssignAssign
Variable_5random_normal_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
l
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
f
random_normal_6/shapeConst*
dtype0*
_output_shapes
:*
valueB"�     
Y
random_normal_6/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_6/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*
T0*
seed2 *

seed *
_output_shapes
:	�
�
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes
:	�
k
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0*
_output_shapes
:	�
�

Variable_6
VariableV2*
dtype0*
shape:	�*
_output_shapes
:	�*
	container *
shared_name 
�
Variable_6/AssignAssign
Variable_6random_normal_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�*
use_locking(*
validate_shape(
p
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
_
random_normal_7/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Y
random_normal_7/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_7/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
dtype0*
T0*
seed2 *

seed *
_output_shapes
:
}
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
T0*
_output_shapes
:
f
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
T0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
shape:*
_output_shapes
:*
	container *
shared_name 
�
Variable_7/AssignAssign
Variable_7random_normal_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
k
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
MatMulMatMulPlaceholderVariable/read*
transpose_a( *
T0*
transpose_b( *(
_output_shapes
:����������
V
AddAddMatMulVariable_1/read*
T0*(
_output_shapes
:����������
D
ReluReluAdd*
T0*(
_output_shapes
:����������
�
MatMul_1MatMulReluVariable_2/read*
transpose_a( *
T0*
transpose_b( *(
_output_shapes
:����������
Z
Add_1AddMatMul_1Variable_3/read*
T0*(
_output_shapes
:����������
H
Relu_1ReluAdd_1*
T0*(
_output_shapes
:����������
�
MatMul_2MatMulRelu_1Variable_4/read*
transpose_a( *
T0*
transpose_b( *(
_output_shapes
:����������
Z
Add_2AddMatMul_2Variable_5/read*
T0*(
_output_shapes
:����������
H
Relu_2ReluAdd_2*
T0*(
_output_shapes
:����������
�
MatMul_3MatMulRelu_2Variable_6/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
Y
Add_3AddMatMul_3Variable_7/read*
T0*'
_output_shapes
:���������
G
Relu_3ReluAdd_3*
T0*'
_output_shapes
:���������
�
MatMul_4MatMulRelu_2Variable_6/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
Y
Add_4AddMatMul_4Variable_7/read*
T0*'
_output_shapes
:���������
F
RankConst*
dtype0*
_output_shapes
: *
value	B :
J
ShapeShapeAdd_4*
out_type0*
T0*
_output_shapes
:
H
Rank_1Const*
dtype0*
_output_shapes
: *
value	B :
L
Shape_1ShapeAdd_4*
out_type0*
T0*
_output_shapes
:
G
Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
:
SubSubRank_1Sub/y*
T0*
_output_shapes
: 
R
Slice/beginPackSub*
N*
T0*
_output_shapes
:*

axis 
T

Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
b
SliceSliceShape_1Slice/begin
Slice/size*
T0*
_output_shapes
:*
Index0
b
concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
q
concatConcatV2concat/values_0Sliceconcat/axis*
N*
T0*
_output_shapes
:*

Tidx0
j
ReshapeReshapeAdd_4concat*
T0*0
_output_shapes
:������������������*
Tshape0
H
Rank_2Const*
dtype0*
_output_shapes
: *
value	B :
T
Shape_2ShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
I
Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
>
Sub_1SubRank_2Sub_1/y*
T0*
_output_shapes
: 
V
Slice_1/beginPackSub_1*
N*
T0*
_output_shapes
:*

axis 
V
Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
h
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
_output_shapes
:*
Index0
d
concat_1/values_0Const*
dtype0*
_output_shapes
:*
valueB:
���������
O
concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
y
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0
v
	Reshape_1ReshapePlaceholder_1concat_1*
T0*0
_output_shapes
:������������������*
Tshape0
�
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogitsReshape	Reshape_1*
T0*?
_output_shapes-
+:���������:������������������
I
Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
<
Sub_2SubRankSub_2/y*
T0*
_output_shapes
: 
W
Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
U
Slice_2/sizePackSub_2*
N*
T0*
_output_shapes
:*

axis 
o
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*#
_output_shapes
:���������*
Index0
x
	Reshape_2ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*#
_output_shapes
:���������*
Tshape0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
\
MeanMean	Reshape_2Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
T
gradients/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients/Mean_grad/ShapeShape	Reshape_2*
out_type0*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
d
gradients/Mean_grad/Shape_1Shape	Reshape_2*
out_type0*
T0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
{
gradients/Reshape_2_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
out_type0*
T0*
_output_shapes
:
�
 gradients/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_2_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
}
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
�
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_2_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
�
0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:������������������
a
gradients/Reshape_grad/ShapeShapeAdd_4*
out_type0*
T0*
_output_shapes
:
�
gradients/Reshape_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
b
gradients/Add_4_grad/ShapeShapeMatMul_4*
out_type0*
T0*
_output_shapes
:
f
gradients/Add_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
*gradients/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_4_grad/Shapegradients/Add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_4_grad/SumSumgradients/Reshape_grad/Reshape*gradients/Add_4_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_4_grad/ReshapeReshapegradients/Add_4_grad/Sumgradients/Add_4_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
gradients/Add_4_grad/Sum_1Sumgradients/Reshape_grad/Reshape,gradients/Add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_4_grad/Reshape_1Reshapegradients/Add_4_grad/Sum_1gradients/Add_4_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
m
%gradients/Add_4_grad/tuple/group_depsNoOp^gradients/Add_4_grad/Reshape^gradients/Add_4_grad/Reshape_1
�
-gradients/Add_4_grad/tuple/control_dependencyIdentitygradients/Add_4_grad/Reshape&^gradients/Add_4_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_4_grad/Reshape*
T0*'
_output_shapes
:���������
�
/gradients/Add_4_grad/tuple/control_dependency_1Identitygradients/Add_4_grad/Reshape_1&^gradients/Add_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_4_grad/Reshape_1*
T0*
_output_shapes
:
�
gradients/MatMul_4_grad/MatMulMatMul-gradients/Add_4_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:����������
�
 gradients/MatMul_4_grad/MatMul_1MatMulRelu_2-gradients/Add_4_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	�
t
(gradients/MatMul_4_grad/tuple/group_depsNoOp^gradients/MatMul_4_grad/MatMul!^gradients/MatMul_4_grad/MatMul_1
�
0gradients/MatMul_4_grad/tuple/control_dependencyIdentitygradients/MatMul_4_grad/MatMul)^gradients/MatMul_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_4_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_4_grad/tuple/control_dependency_1Identity gradients/MatMul_4_grad/MatMul_1)^gradients/MatMul_4_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_4_grad/MatMul_1*
T0*
_output_shapes
:	�
�
gradients/Relu_2_grad/ReluGradReluGrad0gradients/MatMul_4_grad/tuple/control_dependencyRelu_2*
T0*(
_output_shapes
:����������
b
gradients/Add_2_grad/ShapeShapeMatMul_2*
out_type0*
T0*
_output_shapes
:
g
gradients/Add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
gradients/Add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
_output_shapes	
:�*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
�
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_2_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:����������
�
 gradients/MatMul_2_grad/MatMul_1MatMulRelu_1-gradients/Add_2_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:
��
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
�
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
gradients/Relu_1_grad/ReluGradReluGrad0gradients/MatMul_2_grad/tuple/control_dependencyRelu_1*
T0*(
_output_shapes
:����������
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
out_type0*
T0*
_output_shapes
:
g
gradients/Add_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
_output_shapes	
:�*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_1_grad/Reshape*
T0*(
_output_shapes
:����������
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_2/read*
transpose_a( *
T0*
transpose_b(*(
_output_shapes
:����������
�
 gradients/MatMul_1_grad/MatMul_1MatMulRelu-gradients/Add_1_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:
��
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:����������
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
gradients/Relu_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu*
T0*(
_output_shapes
:����������
^
gradients/Add_grad/ShapeShapeMatMul*
out_type0*
T0*
_output_shapes
:
e
gradients/Add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:�
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
�
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
_output_shapes	
:�*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*-
_class#
!loc:@gradients/Add_grad/Reshape*
T0*(
_output_shapes
:����������
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/Add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable/read*
transpose_a( *
T0*
transpose_b(*'
_output_shapes
:���������d
�
gradients/MatMul_grad/MatMul_1MatMulPlaceholder+gradients/Add_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( *
_output_shapes
:	d�
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������d
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	d�
{
beta1_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
_output_shapes
: *
valueB
 *fff?
�
beta1_power
VariableV2*
shape: *
_output_shapes
: *
_class
loc:@Variable*
	container *
dtype0*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
_class
loc:@Variable*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
g
beta1_power/readIdentitybeta1_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
{
beta2_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
_output_shapes
: *
valueB
 *w�?
�
beta2_power
VariableV2*
shape: *
_output_shapes
: *
_class
loc:@Variable*
	container *
dtype0*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@Variable*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
g
beta2_power/readIdentitybeta2_power*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Variable/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable*
_output_shapes
:	d�*
valueB	d�*    
�
Variable/Adam
VariableV2*
shape:	d�*
_output_shapes
:	d�*
_class
loc:@Variable*
dtype0*
shared_name *
	container 
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/Const*
_class
loc:@Variable*
T0*
_output_shapes
:	d�*
use_locking(*
validate_shape(
t
Variable/Adam/readIdentityVariable/Adam*
_class
loc:@Variable*
T0*
_output_shapes
:	d�
�
!Variable/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable*
_output_shapes
:	d�*
valueB	d�*    
�
Variable/Adam_1
VariableV2*
shape:	d�*
_output_shapes
:	d�*
_class
loc:@Variable*
dtype0*
shared_name *
	container 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/Const*
_class
loc:@Variable*
T0*
_output_shapes
:	d�*
use_locking(*
validate_shape(
x
Variable/Adam_1/readIdentityVariable/Adam_1*
_class
loc:@Variable*
T0*
_output_shapes
:	d�
�
!Variable_1/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_1*
_output_shapes	
:�*
valueB�*    
�
Variable_1/Adam
VariableV2*
shape:�*
_output_shapes	
:�*
_class
loc:@Variable_1*
dtype0*
shared_name *
	container 
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/Const*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
v
Variable_1/Adam/readIdentityVariable_1/Adam*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
#Variable_1/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_1*
_output_shapes	
:�*
valueB�*    
�
Variable_1/Adam_1
VariableV2*
shape:�*
_output_shapes	
:�*
_class
loc:@Variable_1*
dtype0*
shared_name *
	container 
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/Const*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
z
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�
�
!Variable_2/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_2* 
_output_shapes
:
��*
valueB
��*    
�
Variable_2/Adam
VariableV2*
shape:
��* 
_output_shapes
:
��*
_class
loc:@Variable_2*
dtype0*
shared_name *
	container 
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/Const*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��*
use_locking(*
validate_shape(
{
Variable_2/Adam/readIdentityVariable_2/Adam*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��
�
#Variable_2/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_2* 
_output_shapes
:
��*
valueB
��*    
�
Variable_2/Adam_1
VariableV2*
shape:
��* 
_output_shapes
:
��*
_class
loc:@Variable_2*
dtype0*
shared_name *
	container 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/Const*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��*
use_locking(*
validate_shape(

Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��
�
!Variable_3/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_3*
_output_shapes	
:�*
valueB�*    
�
Variable_3/Adam
VariableV2*
shape:�*
_output_shapes	
:�*
_class
loc:@Variable_3*
dtype0*
shared_name *
	container 
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/Const*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
v
Variable_3/Adam/readIdentityVariable_3/Adam*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
�
#Variable_3/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_3*
_output_shapes	
:�*
valueB�*    
�
Variable_3/Adam_1
VariableV2*
shape:�*
_output_shapes	
:�*
_class
loc:@Variable_3*
dtype0*
shared_name *
	container 
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/Const*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
z
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�
�
!Variable_4/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_4* 
_output_shapes
:
��*
valueB
��*    
�
Variable_4/Adam
VariableV2*
shape:
��* 
_output_shapes
:
��*
_class
loc:@Variable_4*
dtype0*
shared_name *
	container 
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/Const*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��*
use_locking(*
validate_shape(
{
Variable_4/Adam/readIdentityVariable_4/Adam*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
�
#Variable_4/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_4* 
_output_shapes
:
��*
valueB
��*    
�
Variable_4/Adam_1
VariableV2*
shape:
��* 
_output_shapes
:
��*
_class
loc:@Variable_4*
dtype0*
shared_name *
	container 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/Const*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��*
use_locking(*
validate_shape(

Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��
�
!Variable_5/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_5*
_output_shapes	
:�*
valueB�*    
�
Variable_5/Adam
VariableV2*
shape:�*
_output_shapes	
:�*
_class
loc:@Variable_5*
dtype0*
shared_name *
	container 
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/Const*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
v
Variable_5/Adam/readIdentityVariable_5/Adam*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
�
#Variable_5/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_5*
_output_shapes	
:�*
valueB�*    
�
Variable_5/Adam_1
VariableV2*
shape:�*
_output_shapes	
:�*
_class
loc:@Variable_5*
dtype0*
shared_name *
	container 
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/Const*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�*
use_locking(*
validate_shape(
z
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�
�
!Variable_6/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_6*
_output_shapes
:	�*
valueB	�*    
�
Variable_6/Adam
VariableV2*
shape:	�*
_output_shapes
:	�*
_class
loc:@Variable_6*
dtype0*
shared_name *
	container 
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/Const*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�*
use_locking(*
validate_shape(
z
Variable_6/Adam/readIdentityVariable_6/Adam*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
#Variable_6/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_6*
_output_shapes
:	�*
valueB	�*    
�
Variable_6/Adam_1
VariableV2*
shape:	�*
_output_shapes
:	�*
_class
loc:@Variable_6*
dtype0*
shared_name *
	container 
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/Const*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�*
use_locking(*
validate_shape(
~
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�
�
!Variable_7/Adam/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_7*
_output_shapes
:*
valueB*    
�
Variable_7/Adam
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_7*
dtype0*
shared_name *
	container 
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/Const*
_class
loc:@Variable_7*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
u
Variable_7/Adam/readIdentityVariable_7/Adam*
_class
loc:@Variable_7*
T0*
_output_shapes
:
�
#Variable_7/Adam_1/Initializer/ConstConst*
dtype0*
_class
loc:@Variable_7*
_output_shapes
:*
valueB*    
�
Variable_7/Adam_1
VariableV2*
shape:*
_output_shapes
:*
_class
loc:@Variable_7*
dtype0*
shared_name *
	container 
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/Const*
_class
loc:@Variable_7*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
_class
loc:@Variable_7*
T0*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
_class
loc:@Variable*
T0*
_output_shapes
:	d�*
use_locking( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
_class
loc:@Variable_1*
T0*
_output_shapes	
:�*
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_2*
T0* 
_output_shapes
:
��*
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
_class
loc:@Variable_3*
T0*
_output_shapes	
:�*
use_locking( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_4*
T0* 
_output_shapes
:
��*
use_locking( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
_class
loc:@Variable_5*
T0*
_output_shapes	
:�*
use_locking( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_4_grad/tuple/control_dependency_1*
_class
loc:@Variable_6*
T0*
_output_shapes
:	�*
use_locking( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_4_grad/tuple/control_dependency_1*
_class
loc:@Variable_7*
T0*
_output_shapes
:*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
_class
loc:@Variable*
T0*
_output_shapes
: *
use_locking( *
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
_class
loc:@Variable*
T0*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
_class
loc:@Variable*
T0*
_output_shapes
: *
use_locking( *
validate_shape(
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
c
ArgMaxArgMaxAdd_4ArgMax/dimension*
T0*#
_output_shapes
:���������*

Tidx0
T
ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
o
ArgMax_1ArgMaxPlaceholder_1ArgMax_1/dimension*
T0*#
_output_shapes
:���������*

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:���������
R
Cast_1CastEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
]
Mean_1MeanCast_1Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0"(���