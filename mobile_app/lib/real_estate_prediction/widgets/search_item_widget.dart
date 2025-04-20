import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class SearchItemWidget extends StatelessWidget {
  const SearchItemWidget({
    super.key,
    required this.title,
    required this.items,
    this.selectedIndex,
    this.onSelect,
  });

  final String title;
  final List<String> items;
  final int? selectedIndex;
  final void Function(int index)? onSelect;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      spacing: 8.h,
      children: [
        Text(
          title,
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
            fontFamily: "Roboto",
            fontWeight: FontWeight.w700,
          ),
        ),
        Container(
          width: double.infinity,
          decoration: BoxDecoration(
            color: Color(0xffE8EFFF),
            borderRadius: BorderRadius.circular(8.r),
          ),
          padding: EdgeInsets.all(8.r),
          child: SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: List.generate(items.length, (index) {
                final selected = selectedIndex == index;
                return InkWell(
                  onTap: () {
                    onSelect?.call(index);
                  },
                  child: Container(
                    margin: EdgeInsetsDirectional.only(end: 8.w),
                    decoration: BoxDecoration(
                      color: selected ? Color(0xff5A6CE9) : Color(0xffE8EFFF),
                      borderRadius: BorderRadius.circular(4.r),
                    ),
                    padding: EdgeInsets.symmetric(
                      horizontal: 12.w,
                      vertical: 8.h,
                    ),
                    constraints: BoxConstraints(
                      minWidth: 40.w,
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      items[index],
                      style: Theme.of(context).textTheme.bodyLarge!.copyWith(
                        color: selected ? Colors.white : Color(0xff747474),
                      ),
                    ),
                  ),
                );
              }),
            ),
          ),
        ),
      ],
    );
  }
}
