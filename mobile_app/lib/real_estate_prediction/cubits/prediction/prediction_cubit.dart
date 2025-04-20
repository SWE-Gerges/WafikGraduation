import 'package:dio/dio.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:freezed_annotation/freezed_annotation.dart';

part 'prediction_state.dart';

part 'prediction_cubit.freezed.dart';

class PredictionCubit extends Cubit<PredictionState> {
  PredictionCubit() : super(PredictionState.initial());

  final TextEditingController _sizeController = TextEditingController();

  TextEditingController get sizeController => _sizeController;

  List<String> get propertyTypes => _propertyTypes;
  final List<String> _propertyTypes = [
    "Duplex",
    "Villa",
    "Apartment",
    "Townhouse",
    "Penthouse",
    "iVilla",
    "Twin House",
    "Hotel Apartment",
    "Chalet",
    "Compound",
  ];

  final List<String> _bedrooms = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];

  final List<String> _bathrooms = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];

  void setLocation({required String? location}) {
    emit(state.copyWith(selectedLocation: location, state: States.init));
  }

  void selectPropertyType({required int index}) {
    emit(state.copyWith(selectedPropertyType: index, state: States.init));
  }

  void selectBedroomsCount({required int index}) {
    emit(state.copyWith(selectedBedroomsCount: index, state: States.init));
  }

  void selectBathroomsCount({required int index}) {
    emit(state.copyWith(selectedBathroomsCount: index, state: States.init));
  }

  Future<void> predict() async {
    emit(state.copyWith(state: States.loading));
    try {
      final query = {
        "type": _propertyTypes[state.selectedPropertyType],
        "location": state.selectedLocation,
        "bedrooms": _bedrooms[state.selectedBedroomsCount],
        "bathrooms": _bathrooms[state.selectedBathroomsCount],
        "size_sqm": double.tryParse(_sizeController.text) ?? 0,
      };
      final result = await Dio().get(
        "https://wafiksleim.pythonanywhere.com/mobileApi/predict",
        queryParameters: query,
      );
      final predictedPrice = double.tryParse(
        result.data["predicted_price"].toString(),
      );
      emit(state.copyWith(state: States.loaded, result: predictedPrice));
    } catch (e) {
      emit(state.copyWith(state: States.error, error: e.toString()));
    }
  }

  List<String> get bedrooms => _bedrooms;

  List<String> get bathrooms => _bathrooms;
}
