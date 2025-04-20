import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/core/resources/icons_manager.dart';
import 'package:mobile_app/real_estate_prediction/cubits/locations/locations_cubit.dart';

class LocationsSearchSheet extends StatelessWidget {
  const LocationsSearchSheet({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider(
      create: (context) => LocationsCubit()..init(),
      child: BlocBuilder<LocationsCubit, LocationsState>(
        builder: (context, state) {
          final cubit = context.read<LocationsCubit>();
          return DraggableScrollableSheet(
            maxChildSize: .9,
            initialChildSize: .7,
            minChildSize: .4,
            expand: false,
            snap: true,
            builder: (context, controller) {
              cubit.startListenOnScroll(scrollController: controller);
              return Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadiusDirectional.only(
                    topStart: Radius.circular(8.r),
                    topEnd: Radius.circular(8.r),
                  ),
                ),
                child: SingleChildScrollView(
                  controller: controller,
                  padding: EdgeInsets.symmetric(
                    vertical: 20.h,
                    horizontal: 16.w,
                  ),
                  child: Column(
                    spacing: 18.h,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      TextFormField(
                        controller: cubit.searchFieldController,
                        autofocus: true,
                        onChanged: (value) {
                          cubit.search(key: value);
                        },
                        decoration: InputDecoration(
                          prefixIcon: IconButton(
                            onPressed: () {
                              Navigator.of(context).pop();
                            },
                            icon: Icon(Icons.arrow_back),
                          ),
                          filled: true,
                          fillColor: Color(0xffE8EFFF),
                          suffixIcon: IconButton(
                            onPressed: () {
                              cubit.clear();
                            },
                            icon: Icon(Icons.close),
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8.r),
                            borderSide: BorderSide(
                              width: 1,
                              color: Color(0xffE8EFFF),
                            ),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8.r),
                            borderSide: BorderSide(
                              width: 1,
                              color: Color(0xffE8EFFF),
                            ),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8.r),
                            borderSide: BorderSide(
                              width: 1,
                              color: Color(0xffE8EFFF),
                            ),
                          ),
                        ),
                      ),
                      Column(
                        spacing: 8.h,
                        children:
                            state.locations
                                .map(
                                  (e) => InkWell(
                                    onTap: () {
                                      Navigator.of(context).pop(e);
                                    },
                                    child: Row(
                                      spacing: 16.w,
                                      children: [
                                        SvgPicture.asset(IconsManager.location),
                                        Expanded(
                                          child: Text(
                                            e,
                                            style: Theme.of(
                                              context,
                                            ).textTheme.bodyMedium!.copyWith(
                                              color: Color(0xff49454F),
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                )
                                .toList(),
                      ),
                      if(state.nextPageState == States.loading)
                        Center(child: CircularProgressIndicator(),)
                    ],
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
