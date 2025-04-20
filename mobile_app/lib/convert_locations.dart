import 'dart:convert';
import 'dart:io';




Future<void> main() async {
  final inputPath = '../assets/files/locations.json';

  // Step 1: Read the input JSON file
  final inputFile = File(inputPath);
  final jsonString = await inputFile.readAsString();
  final List<dynamic> originalList = jsonDecode(jsonString);

  // Step 2: Extract only the location strings
  final List<String> locationList = originalList
      .map((item) => item['location']?.toString() ?? '')
      .where((location) => location.isNotEmpty)
      .toList();

  // Step 3: Overwrite the same input file with the new list
  await inputFile.writeAsString(jsonEncode(locationList));

  print('Input file has been overwritten with location list.');
}
