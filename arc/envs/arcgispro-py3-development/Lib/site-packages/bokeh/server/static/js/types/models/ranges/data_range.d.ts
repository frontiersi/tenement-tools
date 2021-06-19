import { Range } from "./range";
import { DataRenderer } from "../renderers/data_renderer";
import * as p from "../../core/properties";
export declare namespace DataRange {
    type Attrs = p.AttrsOf<Props>;
    type Props = Range.Props & {
        /** @deprecated */
        names: p.Property<string[]>;
        renderers: p.Property<DataRenderer[] | "auto">;
    };
}
export interface DataRange extends DataRange.Attrs {
}
export declare abstract class DataRange extends Range {
    properties: DataRange.Props;
    constructor(attrs?: Partial<DataRange.Attrs>);
    static init_DataRange(): void;
}
//# sourceMappingURL=data_range.d.ts.map