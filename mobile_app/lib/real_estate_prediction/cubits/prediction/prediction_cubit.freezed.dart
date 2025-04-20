// dart format width=80
// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'prediction_cubit.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$PredictionState {

 States get state; int get selectedPropertyType; int get selectedBedroomsCount; int get selectedBathroomsCount; String? get selectedLocation; String? get error; double? get result;
/// Create a copy of PredictionState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$PredictionStateCopyWith<PredictionState> get copyWith => _$PredictionStateCopyWithImpl<PredictionState>(this as PredictionState, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is PredictionState&&(identical(other.state, state) || other.state == state)&&(identical(other.selectedPropertyType, selectedPropertyType) || other.selectedPropertyType == selectedPropertyType)&&(identical(other.selectedBedroomsCount, selectedBedroomsCount) || other.selectedBedroomsCount == selectedBedroomsCount)&&(identical(other.selectedBathroomsCount, selectedBathroomsCount) || other.selectedBathroomsCount == selectedBathroomsCount)&&(identical(other.selectedLocation, selectedLocation) || other.selectedLocation == selectedLocation)&&(identical(other.error, error) || other.error == error)&&(identical(other.result, result) || other.result == result));
}


@override
int get hashCode => Object.hash(runtimeType,state,selectedPropertyType,selectedBedroomsCount,selectedBathroomsCount,selectedLocation,error,result);

@override
String toString() {
  return 'PredictionState(state: $state, selectedPropertyType: $selectedPropertyType, selectedBedroomsCount: $selectedBedroomsCount, selectedBathroomsCount: $selectedBathroomsCount, selectedLocation: $selectedLocation, error: $error, result: $result)';
}


}

/// @nodoc
abstract mixin class $PredictionStateCopyWith<$Res>  {
  factory $PredictionStateCopyWith(PredictionState value, $Res Function(PredictionState) _then) = _$PredictionStateCopyWithImpl;
@useResult
$Res call({
 States state, int selectedPropertyType, int selectedBedroomsCount, int selectedBathroomsCount, String? selectedLocation, String? error, double? result
});




}
/// @nodoc
class _$PredictionStateCopyWithImpl<$Res>
    implements $PredictionStateCopyWith<$Res> {
  _$PredictionStateCopyWithImpl(this._self, this._then);

  final PredictionState _self;
  final $Res Function(PredictionState) _then;

/// Create a copy of PredictionState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? state = null,Object? selectedPropertyType = null,Object? selectedBedroomsCount = null,Object? selectedBathroomsCount = null,Object? selectedLocation = freezed,Object? error = freezed,Object? result = freezed,}) {
  return _then(_self.copyWith(
state: null == state ? _self.state : state // ignore: cast_nullable_to_non_nullable
as States,selectedPropertyType: null == selectedPropertyType ? _self.selectedPropertyType : selectedPropertyType // ignore: cast_nullable_to_non_nullable
as int,selectedBedroomsCount: null == selectedBedroomsCount ? _self.selectedBedroomsCount : selectedBedroomsCount // ignore: cast_nullable_to_non_nullable
as int,selectedBathroomsCount: null == selectedBathroomsCount ? _self.selectedBathroomsCount : selectedBathroomsCount // ignore: cast_nullable_to_non_nullable
as int,selectedLocation: freezed == selectedLocation ? _self.selectedLocation : selectedLocation // ignore: cast_nullable_to_non_nullable
as String?,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,result: freezed == result ? _self.result : result // ignore: cast_nullable_to_non_nullable
as double?,
  ));
}

}


/// @nodoc


class _PredictionState implements PredictionState {
  const _PredictionState({required this.state, required this.selectedPropertyType, required this.selectedBedroomsCount, required this.selectedBathroomsCount, required this.selectedLocation, required this.error, required this.result});
  

@override final  States state;
@override final  int selectedPropertyType;
@override final  int selectedBedroomsCount;
@override final  int selectedBathroomsCount;
@override final  String? selectedLocation;
@override final  String? error;
@override final  double? result;

/// Create a copy of PredictionState
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$PredictionStateCopyWith<_PredictionState> get copyWith => __$PredictionStateCopyWithImpl<_PredictionState>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _PredictionState&&(identical(other.state, state) || other.state == state)&&(identical(other.selectedPropertyType, selectedPropertyType) || other.selectedPropertyType == selectedPropertyType)&&(identical(other.selectedBedroomsCount, selectedBedroomsCount) || other.selectedBedroomsCount == selectedBedroomsCount)&&(identical(other.selectedBathroomsCount, selectedBathroomsCount) || other.selectedBathroomsCount == selectedBathroomsCount)&&(identical(other.selectedLocation, selectedLocation) || other.selectedLocation == selectedLocation)&&(identical(other.error, error) || other.error == error)&&(identical(other.result, result) || other.result == result));
}


@override
int get hashCode => Object.hash(runtimeType,state,selectedPropertyType,selectedBedroomsCount,selectedBathroomsCount,selectedLocation,error,result);

@override
String toString() {
  return 'PredictionState(state: $state, selectedPropertyType: $selectedPropertyType, selectedBedroomsCount: $selectedBedroomsCount, selectedBathroomsCount: $selectedBathroomsCount, selectedLocation: $selectedLocation, error: $error, result: $result)';
}


}

/// @nodoc
abstract mixin class _$PredictionStateCopyWith<$Res> implements $PredictionStateCopyWith<$Res> {
  factory _$PredictionStateCopyWith(_PredictionState value, $Res Function(_PredictionState) _then) = __$PredictionStateCopyWithImpl;
@override @useResult
$Res call({
 States state, int selectedPropertyType, int selectedBedroomsCount, int selectedBathroomsCount, String? selectedLocation, String? error, double? result
});




}
/// @nodoc
class __$PredictionStateCopyWithImpl<$Res>
    implements _$PredictionStateCopyWith<$Res> {
  __$PredictionStateCopyWithImpl(this._self, this._then);

  final _PredictionState _self;
  final $Res Function(_PredictionState) _then;

/// Create a copy of PredictionState
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? state = null,Object? selectedPropertyType = null,Object? selectedBedroomsCount = null,Object? selectedBathroomsCount = null,Object? selectedLocation = freezed,Object? error = freezed,Object? result = freezed,}) {
  return _then(_PredictionState(
state: null == state ? _self.state : state // ignore: cast_nullable_to_non_nullable
as States,selectedPropertyType: null == selectedPropertyType ? _self.selectedPropertyType : selectedPropertyType // ignore: cast_nullable_to_non_nullable
as int,selectedBedroomsCount: null == selectedBedroomsCount ? _self.selectedBedroomsCount : selectedBedroomsCount // ignore: cast_nullable_to_non_nullable
as int,selectedBathroomsCount: null == selectedBathroomsCount ? _self.selectedBathroomsCount : selectedBathroomsCount // ignore: cast_nullable_to_non_nullable
as int,selectedLocation: freezed == selectedLocation ? _self.selectedLocation : selectedLocation // ignore: cast_nullable_to_non_nullable
as String?,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,result: freezed == result ? _self.result : result // ignore: cast_nullable_to_non_nullable
as double?,
  ));
}


}

// dart format on
