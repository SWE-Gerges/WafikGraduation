import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/core/resources/icons_manager.dart';
import 'package:mobile_app/real_estate_prediction/cubits/prediction/prediction_cubit.dart';

import '../widgets/result_dialog_widget.dart';
import '../widgets/search_item_widget.dart';
import 'locations_search_sheet.dart';

class RealEstatePredictionFormScreen extends StatefulWidget {
  const RealEstatePredictionFormScreen({super.key});

  @override
  State<RealEstatePredictionFormScreen> createState() =>
      _RealEstatePredictionFormScreenState();
}

class _RealEstatePredictionFormScreenState
    extends State<RealEstatePredictionFormScreen> {
  final _formKey = GlobalKey<FormState>();

  @override
  Widget build(BuildContext context) {
    return BlocConsumer<PredictionCubit, PredictionState>(
      listener: (context, state) {
        if (state.state == States.loaded) {
          showDialog(
            context: context,
            builder: (context) => ResultDialogWidget(),
          );
        }
      },
      builder: (context, state) {
        final cubit = context.read<PredictionCubit>();
        return Scaffold(
          backgroundColor: Color(0xffF3F5F7),
          body: SingleChildScrollView(
            padding: EdgeInsets.symmetric(horizontal: 16.w, vertical: 24.h),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              spacing: 24.h,
              children: [
                SvgPicture.asset(
                  IconsManager.logoSvg,
                  width: 70.w,
                  height: 70.w,
                ),
                InkWell(
                  onTap: () {
                    showModalBottomSheet<String>(
                      context: context,
                      isScrollControlled: true,
                      backgroundColor: Colors.white,
                      builder: (context) => LocationsSearchSheet(),
                    ).then((value) {
                      if (value != null) {
                        cubit.setLocation(location: value);
                      }
                    });
                  },
                  child: Container(
                    width: double.infinity,
                    height: 56.h,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(8.r),
                    ),
                    padding: EdgeInsets.all(8.r),
                    child: Row(
                      spacing: 8.w,
                      children: [
                        Icon(Icons.search),
                        Expanded(
                          child: Text(
                            state.selectedLocation ??
                                "Enter an address, neighborhood, city",
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12.r),
                    boxShadow: [
                      BoxShadow(
                        color: Color(0x0000001A),
                        blurRadius: 10.r,
                        spreadRadius: 0,
                        offset: Offset(0, 4.r),
                      ),
                    ],
                  ),
                  padding: EdgeInsets.symmetric(
                    vertical: 16.h,
                    horizontal: 20.w,
                  ),
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      spacing: 18.h,
                      children: [
                        SearchItemWidget(
                          title: "Type",
                          items: cubit.propertyTypes,
                          selectedIndex: state.selectedPropertyType,
                          onSelect: (index) {
                            cubit.selectPropertyType(index: index);
                          },
                        ),
                        SearchItemWidget(
                          title: "Bedrooms",
                          items: cubit.bedrooms,
                          selectedIndex: state.selectedBedroomsCount,
                          onSelect: (index) {
                            cubit.selectBedroomsCount(index: index);
                          },
                        ),
                        SearchItemWidget(
                          title: "Bathrooms",
                          items: cubit.bathrooms,
                          selectedIndex: state.selectedBathroomsCount,
                          onSelect: (index) {
                            cubit.selectBathroomsCount(index: index);
                          },
                        ),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          spacing: 8.h,
                          children: [
                            Text(
                              "Size",
                              style: Theme.of(
                                context,
                              ).textTheme.bodyLarge?.copyWith(
                                fontFamily: "Roboto",
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            TextFormField(
                              keyboardType: TextInputType.number,
                              inputFormatters: [
                                FilteringTextInputFormatter.digitsOnly,
                              ],
                              validator: (value) {
                                if (value == null || value.isEmpty) {
                                  return 'Please enter a number';
                                }
                                return null;
                              },
                              controller: cubit.sizeController,
                              decoration: InputDecoration(
                                filled: true,
                                labelText: "Size sqm",
                                fillColor: Color(0xffFAFAFA),
                                suffixIconConstraints: BoxConstraints(
                                  maxHeight: 40.w,
                                  maxWidth: 40.w,
                                ),
                                suffixIcon: Container(
                                  margin: EdgeInsets.symmetric(horizontal: 4.w),
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(4.r),
                                    color: Color(0xffF0F0F0),
                                  ),
                                  padding: EdgeInsets.symmetric(
                                    horizontal: 8.w,
                                    vertical: 8.h,
                                  ),
                                  child: Text("M"),
                                ),
                              ),
                            ),
                          ],
                        ),
                        if (state.state == States.loading) ...[
                          Center(child: CircularProgressIndicator()),
                        ] else ...[
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              onPressed: () {
                                if (_formKey.currentState!.validate()) {
                                  cubit.predict();
                                }
                              },
                              child: Text(
                                "Predict",
                                style: Theme.of(context).textTheme.titleLarge!
                                    .copyWith(color: Colors.white),
                              ),
                            ),
                          ),
                        ],
                      ],
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
