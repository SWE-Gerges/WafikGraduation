part of 'prediction_cubit.dart';

enum States { init, loading, loaded, error }

@freezed
sealed class PredictionState with _$PredictionState {
  const factory PredictionState({
    required States state,
    required int selectedPropertyType,
    required int selectedBedroomsCount,
    required int selectedBathroomsCount,
    required String? selectedLocation,
    required String? error,
    required double? result,
  }) = _PredictionState;

  factory PredictionState.initial() => const PredictionState(
    state: States.init,
    selectedPropertyType: 0,
    selectedBedroomsCount: 0,
    selectedBathroomsCount: 0,
    selectedLocation: null,
    error: null,
    result: null,
  );
}
