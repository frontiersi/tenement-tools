import { Range } from "./range";
import { DataRenderer } from "../renderers/data_renderer";
export class DataRange extends Range {
    constructor(attrs) {
        super(attrs);
    }
    static init_DataRange() {
        this.define(({ String, Array, Ref }) => ({
            names: [Array(String), []],
            renderers: [Array(Ref(DataRenderer)), []],
        }));
    }
}
DataRange.__name__ = "DataRange";
DataRange.init_DataRange();
//# sourceMappingURL=data_range.js.map