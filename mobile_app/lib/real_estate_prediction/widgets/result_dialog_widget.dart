import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:gap/gap.dart';
import 'package:intl/intl.dart';
import 'package:mobile_app/core/resources/icons_manager.dart';
import 'package:mobile_app/real_estate_prediction/cubits/prediction/prediction_cubit.dart';

class ResultDialogWidget extends StatelessWidget {
  const ResultDialogWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<PredictionCubit, PredictionState>(
      builder: (context, state) {
        final cubit = context.read<PredictionCubit>();
        return Dialog(
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24.r),
          ),
          insetPadding: EdgeInsets.symmetric(horizontal: 24.w, vertical: 24.h),
          alignment: AlignmentDirectional.center,
          child: Container(
            padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 16.h),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SvgPicture.asset(
                  IconsManager.location,
                  width: 40.w,
                  height: 40.w,
                ),
                Gap(12.h),
                Text(
                  cubit.propertyTypes[state.selectedPropertyType],
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 18.sp,
                    color: Color(0xff515151),
                  ),
                ),
                Gap(18.h),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  spacing: 8.w,
                  children: [
                    SvgPicture.asset(
                      IconsManager.locationMarker,
                      width: 24.w,
                      height: 24.w,
                    ),
                    Expanded(
                      child: Text(
                        state.selectedLocation!,
                        style: Theme.of(context).textTheme.bodyLarge!.copyWith(
                          color: Color(0xff515151),
                        ),
                      ),
                    ),
                  ],
                ),
                Gap(16.h),
                Row(
                  spacing: 2.w,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        spacing: 4.h,
                        children: [
                          Row(
                            spacing: 4.w,
                            children: [
                              SvgPicture.asset(
                                IconsManager.bedroom,
                                width: 18.w,
                                height: 18.w,
                              ),
                              Text(
                                "Bedrooms",
                                style: TextStyle(
                                  fontWeight: FontWeight.w400,
                                  fontSize: 14.sp,
                                  color: Color(0xff747474),
                                ),
                              ),
                            ],
                          ),
                          Text(
                            cubit.bedrooms[state.selectedBedroomsCount],
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 16.sp,
                              color: Color(0xff515151),
                            ),
                          )
                        ],
                      ),
                    ),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        spacing: 4.h,
                        children: [
                          Row(
                            spacing: 4.w,
                            children: [
                              SvgPicture.asset(
                                IconsManager.bathroom,
                                width: 18.w,
                                height: 18.w,
                              ),
                              Text(
                                "Bathrooms",
                                style: TextStyle(
                                  fontWeight: FontWeight.w400,
                                  fontSize: 14.sp,
                                  color: Color(0xff747474),
                                ),
                              ),
                            ],
                          ),
                          Text(
                            cubit.bathrooms[state.selectedBathroomsCount],
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 16.sp,
                              color: Color(0xff515151),
                            ),
                          )
                        ],
                      ),
                    ),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        spacing: 4.h,
                        children: [
                          Row(
                            spacing: 4.w,
                            children: [
                              SvgPicture.asset(
                                IconsManager.bedroom,
                                width: 18.w,
                                height: 18.w,
                              ),
                              Text(
                                "Size",
                                style: TextStyle(
                                  fontWeight: FontWeight.w400,
                                  fontSize: 14.sp,
                                  color: Color(0xff747474),
                                ),
                              ),
                            ],
                          ),
                          Text(
                            cubit.sizeController.text,
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 16.sp,
                              color: Color(0xff515151),
                            ),
                          )
                        ],
                      ),
                    ),
                  ],
                ),
                Gap(8.h),
                Divider(
                  color: Color(0xA6AEC099),
                ),
                Gap(16.h),
                Text(
                  "Price: ${NumberFormat("#,##0.00", "en_US").format(state.result)} EGP",
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    color: Color(0xff5A6CE9),
                    fontSize: 18.sp,
                  ),
                ),
                Gap(4.h),
                Text(
                  "According to our algorithm and the dataset used",
                  style: TextStyle(
                    color: Color(0xff747474)
                  ),
                ),
                Gap(24.h),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: Text(
                      "Done",
                      style: Theme.of(context).textTheme.titleLarge!
                          .copyWith(color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}
