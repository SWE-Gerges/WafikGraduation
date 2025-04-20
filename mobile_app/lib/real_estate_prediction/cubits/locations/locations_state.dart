part of 'locations_cubit.dart';

enum States { init, loading, loaded, error }

@freezed
sealed class LocationsState with _$LocationsState {
  const factory LocationsState({
    required States state,
    required States nextPageState,
    required List<String> locations,
    required String? error,
    required bool isLastPage,
  }) = _LocationsState;

  factory LocationsState.initial() => const LocationsState(
    state: States.init,
    error: null,
    nextPageState: States.init,
    locations: [],
    isLastPage: false,
  );
}
