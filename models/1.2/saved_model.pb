ä|
¤ű
9
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
0
Square
x"T
y"T"
Ttype:
	2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.3.02v1.3.0-rc2-20-g0787eeeÎe
P
PlaceholderPlaceholder*
dtype0*
shape:*
_output_shapes
:
R
Placeholder_1Placeholder*
dtype0*
shape:*
_output_shapes
:
[
Variable/initial_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
˘
Variable/AssignAssignVariableVariable/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes
: 
a
Variable/readIdentityVariable*
T0*
_class
loc:@Variable*
_output_shapes
: 
]
Variable_1/initial_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
n

Variable_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_output_shapes
: 
Ş
Variable_1/AssignAssign
Variable_1Variable_1/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
: 
g
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1*
_output_shapes
: 
I
mulMulVariable/readPlaceholder*
T0*
_output_shapes
:
C
addAddmulVariable_1/read*
T0*
_output_shapes
:
A
subSubaddPlaceholder_1*
T0*
_output_shapes
:
8
SquareSquaresub*
T0*
_output_shapes
:
5
RankRankSquare*
T0*
_output_shapes
: 
M
range/startConst*
value	B : *
dtype0*
_output_shapes
: 
M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
_
rangeRangerange/startRankrange/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
SumSumSquarerange*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
2
initNoOp^Variable/Assign^Variable_1/Assign
[
gradients/ShapeShapeSum*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
[
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
:
g
gradients/Sum_grad/ShapeShapeSquare*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
gradients/Sum_grad/SizeSizegradients/Sum_grad/Shape*
T0*
out_type0*
_output_shapes
: 
k
gradients/Sum_grad/addAddrangegradients/Sum_grad/Size*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
gradients/Sum_grad/Shape_1Shapegradients/Sum_grad/mod*
T0*
out_type0*
_output_shapes
:
`
gradients/Sum_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
`
gradients/Sum_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ť
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
gradients/Sum_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Í
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
N*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
gradients/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/Sum_grad/ReshapeReshapegradients/Fill gradients/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:

gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
T0*

Tmultiples0*
_output_shapes
:
z
gradients/Square_grad/mul/xConst^gradients/Sum_grad/Tile*
valueB
 *   @*
dtype0*
_output_shapes
: 
e
gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
T0*
_output_shapes
:
y
gradients/Square_grad/mul_1Mulgradients/Sum_grad/Tilegradients/Square_grad/mul*
T0*
_output_shapes
:
d
gradients/sub_grad/ShapeShapeadd*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
gradients/sub_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
¤
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¨
gradients/sub_grad/Sum_1Sumgradients/Square_grad/mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ë
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*
_output_shapes
:
Ń
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
_output_shapes
:
d
gradients/add_grad/ShapeShapemul*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
gradients/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
´
gradients/add_grad/SumSum+gradients/sub_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:
¸
gradients/add_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ë
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape*
_output_shapes
:
Ď
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
: 
[
gradients/mul_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
n
gradients/mul_grad/Shape_1ShapePlaceholder*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
´
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
z
gradients/mul_grad/mulMul+gradients/add_grad/tuple/control_dependencyPlaceholder*
T0*
_output_shapes
:

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
~
gradients/mul_grad/mul_1MulVariable/read+gradients/add_grad/tuple/control_dependency*
T0*
_output_shapes
:
Ľ
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
É
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*
_output_shapes
: 
Ń
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*
_output_shapes
:
b
GradientDescent/learning_rateConst*
valueB
 *
×#<*
dtype0*
_output_shapes
: 
ó
4GradientDescent/update_Variable/ApplyGradientDescentApplyGradientDescentVariableGradientDescent/learning_rate+gradients/mul_grad/tuple/control_dependency*
T0*
use_locking( *
_class
loc:@Variable*
_output_shapes
: 
ű
6GradientDescent/update_Variable_1/ApplyGradientDescentApplyGradientDescent
Variable_1GradientDescent/learning_rate-gradients/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
_class
loc:@Variable_1*
_output_shapes
: 

GradientDescentNoOp5^GradientDescent/update_Variable/ApplyGradientDescent7^GradientDescent/update_Variable_1/ApplyGradientDescent

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_6f9e18c478b14ba18e55de03326c5907/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
u
save/SaveV2/tensor_namesConst*)
value BBVariableB
Variable_1*
dtype0*
_output_shapes
:
g
save/SaveV2/shape_and_slicesConst*
valueBB B *
dtype0*
_output_shapes
:

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
T0*

axis *
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
l
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:

save/AssignAssignVariablesave/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable*
_output_shapes
: 
p
save/RestoreV2_1/tensor_namesConst*
valueBB
Variable_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:

save/Assign_1Assign
Variable_1save/RestoreV2_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1*
_output_shapes
: 
8
save/restore_shardNoOp^save/Assign^save/Assign_1
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"
trainable_variableshf
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0"u
	variableshf
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0"
train_op

GradientDescent"$
legacy_init_op

legacy_init_op*m
predict_output[

x
Placeholder:0!
scores
Placeholder_1:0tensorflow/serving/predict