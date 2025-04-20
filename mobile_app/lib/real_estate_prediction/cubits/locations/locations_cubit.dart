import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import 'package:mobile_app/core/services/locations_services.dart';

part 'locations_state.dart';

part 'locations_cubit.freezed.dart';

class LocationsCubit extends Cubit<LocationsState> {
  LocationsCubit() : super(LocationsState.initial());

  final _locationsServices = LocationsServices();

  final _searchFieldController = TextEditingController();


  TextEditingController get searchFieldController => _searchFieldController;

  Future<void> init() async {
    emit(state.copyWith(state: States.loading));
    try {
      await _locationsServices.init();
      emit(state.copyWith(state: States.loaded));
    } catch (e) {
      emit(state.copyWith(state: States.error, error: e.toString()));
    }
  }

  Timer? _searchTimer;

  final _pageSize = 30;

  int _currentPage = 0;
  String _key = "";

  void search({required String key}) {
    if (_searchTimer != null) _searchTimer?.cancel();
    _searchTimer = Timer(Duration(milliseconds: 500), () {
      try {
        _currentPage = 0;
        _key = "";
        emit(state.copyWith(state: States.loading, locations: []));
        final result = _locationsServices.search(
          query: key,
          page: _currentPage,
          pageSize: _pageSize,
        );
        emit(
          state.copyWith(
            state: States.loaded,
            locations: result,
            isLastPage: result.length < _pageSize,
          ),
        );
      } catch (e) {
        emit(
          state.copyWith(state: States.error, error: "Something wont wrong"),
        );
      }
    });
  }

  void startListenOnScroll({required ScrollController scrollController}) async {
    scrollController.addListener(() {
      if (scrollController.offset >=
              scrollController.position.maxScrollExtent &&
          state.nextPageState != States.loading) {
        _getNextPage();
      }
    });
  }

  void _getNextPage() {
    emit(state.copyWith(nextPageState: States.loading));
    try {
      final result = _locationsServices.search(
        query: _key,
        page: _currentPage++,
        pageSize: _pageSize,
      );
      final locations = [...state.locations, ...result];
      emit(state.copyWith(
        nextPageState: States.loaded,
        locations: locations,
        isLastPage: result.length < _pageSize,
      ));
    } catch (e) {
      emit(state.copyWith(
        nextPageState: States.error,
        error: "Something wont wrong",
      ));
    }
  }

  void clear() {
    _searchFieldController.text = "";
    emit(state.copyWith(
      locations: [],
      state: States.init,
      nextPageState: States.init,
    ));
  }
}
