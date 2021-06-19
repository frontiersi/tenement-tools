import { ZoomBaseTool, ZoomBaseToolView } from "./zoom_base_tool";
export declare class ZoomOutToolView extends ZoomBaseToolView {
    model: ZoomBaseTool;
}
export interface ZoomOutTool extends ZoomBaseTool.Attrs {
}
export declare class ZoomOutTool extends ZoomBaseTool {
    properties: ZoomBaseTool.Props;
    __view_type__: ZoomBaseToolView;
    constructor(attrs?: Partial<ZoomBaseTool.Attrs>);
    static init_ZoomOutTool(): void;
    sign: -1;
    tool_name: string;
    icon: string;
}
//# sourceMappingURL=zoom_out_tool.d.ts.map