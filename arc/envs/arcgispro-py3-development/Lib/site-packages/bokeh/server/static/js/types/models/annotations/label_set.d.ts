import { TextAnnotation, TextAnnotationView } from "./text_annotation";
import { ColumnarDataSource } from "../sources/columnar_data_source";
import * as mixins from "../../core/property_mixins";
import * as visuals from "../../core/visuals";
import { SpatialUnits } from "../../core/enums";
import * as p from "../../core/properties";
import { Size } from "../../core/layout";
import { Arrayable, FloatArray } from "../../core/types";
import { Context2d } from "../../core/util/canvas";
export declare class LabelSetView extends TextAnnotationView {
    model: LabelSet;
    visuals: LabelSet.Visuals;
    protected _x: FloatArray;
    protected _y: FloatArray;
    protected text: p.Uniform<string>;
    protected angle: p.Uniform<number>;
    protected x_offset: p.Uniform<number>;
    protected y_offset: p.Uniform<number>;
    set_data(source: ColumnarDataSource): void;
    initialize(): void;
    connect_signals(): void;
    protected _calculate_text_dimensions(ctx: Context2d, text: string): [number, number];
    protected _map_data(): [Arrayable<number>, Arrayable<number>];
    protected _render(): void;
    protected _get_size(): Size;
    protected _v_canvas_text(ctx: Context2d, i: number, text: string, sx: number, sy: number, angle: number): void;
    protected _v_css_text(ctx: Context2d, i: number, text: string, sx: number, sy: number, angle: number): void;
}
export declare namespace LabelSet {
    type Attrs = p.AttrsOf<Props>;
    type Props = TextAnnotation.Props & {
        x: p.XCoordinateSpec;
        y: p.YCoordinateSpec;
        x_units: p.Property<SpatialUnits>;
        y_units: p.Property<SpatialUnits>;
        text: p.StringSpec;
        angle: p.AngleSpec;
        x_offset: p.NumberSpec;
        y_offset: p.NumberSpec;
        source: p.Property<ColumnarDataSource>;
    } & Mixins;
    type Mixins = mixins.TextVector & mixins.Prefixed<"border", mixins.LineVector> & mixins.Prefixed<"background", mixins.FillVector>;
    type Visuals = TextAnnotation.Visuals & {
        text: visuals.TextVector;
        border_line: visuals.LineVector;
        background_fill: visuals.FillVector;
    };
}
export interface LabelSet extends LabelSet.Attrs {
}
export declare class LabelSet extends TextAnnotation {
    properties: LabelSet.Props;
    __view_type__: LabelSetView;
    constructor(attrs?: Partial<LabelSet.Attrs>);
    static init_LabelSet(): void;
}
//# sourceMappingURL=label_set.d.ts.map