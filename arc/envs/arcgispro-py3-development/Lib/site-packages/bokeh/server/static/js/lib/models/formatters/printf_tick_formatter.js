import { TickFormatter } from "./tick_formatter";
import { sprintf } from "../../core/util/templating";
export class PrintfTickFormatter extends TickFormatter {
    constructor(attrs) {
        super(attrs);
    }
    static init_PrintfTickFormatter() {
        this.define(({ String }) => ({
            format: [String, "%s"],
        }));
    }
    doFormat(ticks, _opts) {
        return ticks.map((tick) => sprintf(this.format, tick));
    }
}
PrintfTickFormatter.__name__ = "PrintfTickFormatter";
PrintfTickFormatter.init_PrintfTickFormatter();
//# sourceMappingURL=printf_tick_formatter.js.map