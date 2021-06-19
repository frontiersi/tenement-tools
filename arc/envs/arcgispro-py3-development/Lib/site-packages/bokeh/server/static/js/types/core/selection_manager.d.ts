import { HasProps } from "./has_props";
import { Geometry } from "./geometry";
import { SelectionMode } from "./enums";
import { Selection } from "../models/selections/selection";
import type { ColumnarDataSource } from "../models/sources/columnar_data_source";
import { DataRenderer, DataRendererView } from "../models/renderers/data_renderer";
import * as p from "./properties";
export declare namespace SelectionManager {
    type Props = HasProps.Props & {
        source: p.Property<ColumnarDataSource>;
    };
    type Attrs = p.AttrsOf<Props>;
}
export interface SelectionManager extends SelectionManager.Attrs {
}
export declare class SelectionManager extends HasProps {
    properties: SelectionManager.Props;
    constructor(attrs?: Partial<SelectionManager.Attrs>);
    static init_SelectionManager(): void;
    inspectors: Map<DataRenderer, Selection>;
    select(renderer_views: DataRendererView[], geometry: Geometry, final: boolean, mode?: SelectionMode): boolean;
    inspect(renderer_view: DataRendererView, geometry: Geometry): boolean;
    clear(rview?: DataRendererView): void;
    get_or_create_inspector(renderer: DataRenderer): Selection;
}
//# sourceMappingURL=selection_manager.d.ts.map