import 'dart:convert';
import 'dart:isolate';
import 'package:flutter/services.dart';

final class LocationsServices {
  LocationsServices();

  static const _jsonPath = 'assets/files/locations.json';

  List<String> _locations = [];
  bool _loaded = false;

  /// Loads and parses the JSON file in an isolate
  Future<void> init() async {
    if (_loaded) return;

    // Load raw string from assets
    final jsonString = await rootBundle.loadString(_jsonPath);

    // Use isolate to parse
    _locations = await _parseJsonInIsolate(jsonString);
    _loaded = true;
  }

  /// Searches locations with pagination
  List<String> search({
    required String query,
    int page = 0,
    int pageSize = 10,
  }) {
    final lowerQuery = query.toLowerCase();
    final filtered = _locations.where(
          (loc) => loc.toLowerCase().contains(lowerQuery),
    ).toList();

    final start = page * pageSize;
    final end = (start + pageSize).clamp(0, filtered.length);

    if (start >= filtered.length) return [];
    return filtered.sublist(start, end);
  }

  int getTotalPages(String query, int pageSize) {
    final lowerQuery = query.toLowerCase();
    final count = _locations.where((loc) => loc.toLowerCase().contains(lowerQuery)).length;
    return (count / pageSize).ceil();
  }

  List<String> get allLocations => _locations;
}

Future<List<String>> _parseJsonInIsolate(String jsonString) async {
  final response = ReceivePort();

  await Isolate.spawn(_decodeJsonIsolate, [response.sendPort, jsonString]);

  return await response.first as List<String>;
}

void _decodeJsonIsolate(List<dynamic> message) {
  final SendPort sendPort = message[0];
  final String jsonString = message[1];

  final List<dynamic> decoded = jsonDecode(jsonString);
  final List<String> locations = List<String>.from(decoded);
  sendPort.send(locations);
}
