import tz from "timezone";
import { AbstractSlider, AbstractRangeSliderView } from "./abstract_slider";
import { isString } from "../../core/util/types";
export class DateRangeSliderView extends AbstractRangeSliderView {
}
DateRangeSliderView.__name__ = "DateRangeSliderView";
export class DateRangeSlider extends AbstractSlider {
    constructor(attrs) {
        super(attrs);
        this.behaviour = "drag";
        this.connected = [false, true, false];
    }
    static init_DateRangeSlider() {
        this.prototype.default_view = DateRangeSliderView;
        this.override({
            format: "%d %b %Y",
        });
    }
    _formatter(value, format) {
        if (isString(format))
            return tz(value, format);
        else
            return format.compute(value);
    }
}
DateRangeSlider.__name__ = "DateRangeSlider";
DateRangeSlider.init_DateRangeSlider();
//# sourceMappingURL=date_range_slider.js.map