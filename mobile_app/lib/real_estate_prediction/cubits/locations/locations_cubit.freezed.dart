// dart format width=80
// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'locations_cubit.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;
/// @nodoc
mixin _$LocationsState {

 States get state; States get nextPageState; List<String> get locations; String? get error; bool get isLastPage;
/// Create a copy of LocationsState
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$LocationsStateCopyWith<LocationsState> get copyWith => _$LocationsStateCopyWithImpl<LocationsState>(this as LocationsState, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is LocationsState&&(identical(other.state, state) || other.state == state)&&(identical(other.nextPageState, nextPageState) || other.nextPageState == nextPageState)&&const DeepCollectionEquality().equals(other.locations, locations)&&(identical(other.error, error) || other.error == error)&&(identical(other.isLastPage, isLastPage) || other.isLastPage == isLastPage));
}


@override
int get hashCode => Object.hash(runtimeType,state,nextPageState,const DeepCollectionEquality().hash(locations),error,isLastPage);

@override
String toString() {
  return 'LocationsState(state: $state, nextPageState: $nextPageState, locations: $locations, error: $error, isLastPage: $isLastPage)';
}


}

/// @nodoc
abstract mixin class $LocationsStateCopyWith<$Res>  {
  factory $LocationsStateCopyWith(LocationsState value, $Res Function(LocationsState) _then) = _$LocationsStateCopyWithImpl;
@useResult
$Res call({
 States state, States nextPageState, List<String> locations, String? error, bool isLastPage
});




}
/// @nodoc
class _$LocationsStateCopyWithImpl<$Res>
    implements $LocationsStateCopyWith<$Res> {
  _$LocationsStateCopyWithImpl(this._self, this._then);

  final LocationsState _self;
  final $Res Function(LocationsState) _then;

/// Create a copy of LocationsState
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? state = null,Object? nextPageState = null,Object? locations = null,Object? error = freezed,Object? isLastPage = null,}) {
  return _then(_self.copyWith(
state: null == state ? _self.state : state // ignore: cast_nullable_to_non_nullable
as States,nextPageState: null == nextPageState ? _self.nextPageState : nextPageState // ignore: cast_nullable_to_non_nullable
as States,locations: null == locations ? _self.locations : locations // ignore: cast_nullable_to_non_nullable
as List<String>,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,isLastPage: null == isLastPage ? _self.isLastPage : isLastPage // ignore: cast_nullable_to_non_nullable
as bool,
  ));
}

}


/// @nodoc


class _LocationsState implements LocationsState {
  const _LocationsState({required this.state, required this.nextPageState, required final  List<String> locations, required this.error, required this.isLastPage}): _locations = locations;
  

@override final  States state;
@override final  States nextPageState;
 final  List<String> _locations;
@override List<String> get locations {
  if (_locations is EqualUnmodifiableListView) return _locations;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_locations);
}

@override final  String? error;
@override final  bool isLastPage;

/// Create a copy of LocationsState
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$LocationsStateCopyWith<_LocationsState> get copyWith => __$LocationsStateCopyWithImpl<_LocationsState>(this, _$identity);



@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _LocationsState&&(identical(other.state, state) || other.state == state)&&(identical(other.nextPageState, nextPageState) || other.nextPageState == nextPageState)&&const DeepCollectionEquality().equals(other._locations, _locations)&&(identical(other.error, error) || other.error == error)&&(identical(other.isLastPage, isLastPage) || other.isLastPage == isLastPage));
}


@override
int get hashCode => Object.hash(runtimeType,state,nextPageState,const DeepCollectionEquality().hash(_locations),error,isLastPage);

@override
String toString() {
  return 'LocationsState(state: $state, nextPageState: $nextPageState, locations: $locations, error: $error, isLastPage: $isLastPage)';
}


}

/// @nodoc
abstract mixin class _$LocationsStateCopyWith<$Res> implements $LocationsStateCopyWith<$Res> {
  factory _$LocationsStateCopyWith(_LocationsState value, $Res Function(_LocationsState) _then) = __$LocationsStateCopyWithImpl;
@override @useResult
$Res call({
 States state, States nextPageState, List<String> locations, String? error, bool isLastPage
});




}
/// @nodoc
class __$LocationsStateCopyWithImpl<$Res>
    implements _$LocationsStateCopyWith<$Res> {
  __$LocationsStateCopyWithImpl(this._self, this._then);

  final _LocationsState _self;
  final $Res Function(_LocationsState) _then;

/// Create a copy of LocationsState
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? state = null,Object? nextPageState = null,Object? locations = null,Object? error = freezed,Object? isLastPage = null,}) {
  return _then(_LocationsState(
state: null == state ? _self.state : state // ignore: cast_nullable_to_non_nullable
as States,nextPageState: null == nextPageState ? _self.nextPageState : nextPageState // ignore: cast_nullable_to_non_nullable
as States,locations: null == locations ? _self._locations : locations // ignore: cast_nullable_to_non_nullable
as List<String>,error: freezed == error ? _self.error : error // ignore: cast_nullable_to_non_nullable
as String?,isLastPage: null == isLastPage ? _self.isLastPage : isLastPage // ignore: cast_nullable_to_non_nullable
as bool,
  ));
}


}

// dart format on
