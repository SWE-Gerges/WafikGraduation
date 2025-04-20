import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:mobile_app/real_estate_prediction/cubits/prediction/prediction_cubit.dart';
import 'package:mobile_app/real_estate_prediction/screens/real_estate_prediction_form_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => PredictionCubit(),
      child: ScreenUtilInit(
        designSize: const Size(375, 812),
        minTextAdapt: true,
        splitScreenMode: true,
        builder:
            (context, _) => MaterialApp(
              debugShowCheckedModeBanner: false,
              title: 'Flutter Demo',
              theme: ThemeData(
                colorScheme: ColorScheme.fromSeed(seedColor: Color(0xff5A6CE9)),
                inputDecorationTheme: InputDecorationTheme(
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8.r),
                    borderSide: BorderSide(width: 1, color: Color(0xffFAFAFA)),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8.r),
                    borderSide: BorderSide(width: 1, color: Color(0xffFAFAFA)),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8.r),
                    borderSide: BorderSide(width: 1, color: Color(0xffFAFAFA)),
                  ),
                ),
                elevatedButtonTheme: ElevatedButtonThemeData(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Color(0xff5A6CE9),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8.r),
                    ),
                    textStyle: Theme.of(context)
                        .textTheme
                        .titleLarge!
                        .copyWith(color: Colors.white),
                  )
                ),
                textTheme: TextTheme(
                  displayLarge: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 57.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  displayMedium: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 45.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  displaySmall: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 36.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  headlineLarge: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 32.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  headlineMedium: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 28.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  headlineSmall: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 24.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  titleLarge: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 22.sp,
                    fontWeight: FontWeight.w500,
                  ),
                  titleMedium: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 16.sp,
                    fontWeight: FontWeight.w500,
                  ),
                  titleSmall: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 14.sp,
                    fontWeight: FontWeight.w500,
                  ),
                  bodyLarge: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 16.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  bodyMedium: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 14.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  bodySmall: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 12.sp,
                    fontWeight: FontWeight.normal,
                  ),
                  labelLarge: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 14.sp,
                    fontWeight: FontWeight.w500,
                  ),
                  labelMedium: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 12.sp,
                    fontWeight: FontWeight.w500,
                  ),
                  labelSmall: TextStyle(
                    fontFamily: 'Roboto',
                    fontSize: 11.sp,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              home: const RealEstatePredictionFormScreen(),
            ),
      ),
    );
  }
}
