import { ActionTool, ActionToolView } from "./action_tool";
import { Dimensions } from "../../../core/enums";
import * as p from "../../../core/properties";
export declare abstract class ZoomBaseToolView extends ActionToolView {
    model: ZoomBaseTool;
    doit(): void;
}
export declare namespace ZoomBaseTool {
    type Attrs = p.AttrsOf<Props>;
    type Props = ActionTool.Props & {
        factor: p.Property<number>;
        dimensions: p.Property<Dimensions>;
    };
}
export interface ZoomBaseTool extends ZoomBaseTool.Attrs {
}
export declare abstract class ZoomBaseTool extends ActionTool {
    properties: ZoomBaseTool.Props;
    __view_type__: ZoomBaseToolView;
    constructor(attrs?: Partial<ZoomBaseTool.Attrs>);
    static init_ZoomBaseTool(): void;
    readonly sign: -1 | 1;
    get tooltip(): string;
}
//# sourceMappingURL=zoom_base_tool.d.ts.map