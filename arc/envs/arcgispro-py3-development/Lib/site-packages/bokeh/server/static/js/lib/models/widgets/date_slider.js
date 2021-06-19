import tz from "timezone";
import { AbstractSlider, AbstractSliderView } from "./abstract_slider";
import { isString } from "../../core/util/types";
export class DateSliderView extends AbstractSliderView {
}
DateSliderView.__name__ = "DateSliderView";
export class DateSlider extends AbstractSlider {
    constructor(attrs) {
        super(attrs);
        this.behaviour = "tap";
        this.connected = [true, false];
    }
    static init_DateSlider() {
        this.prototype.default_view = DateSliderView;
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
DateSlider.__name__ = "DateSlider";
DateSlider.init_DateSlider();
//# sourceMappingURL=date_slider.js.map