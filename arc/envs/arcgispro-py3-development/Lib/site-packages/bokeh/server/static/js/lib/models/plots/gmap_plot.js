import { Plot } from "./plot";
import { Model } from "../../model";
import { Range1d } from '../ranges/range1d';
import { GMapPlotView } from "./gmap_plot_canvas";
export { GMapPlotView };
export class MapOptions extends Model {
    constructor(attrs) {
        super(attrs);
    }
    static init_MapOptions() {
        this.define(({ Int, Number }) => ({
            lat: [Number],
            lng: [Number],
            zoom: [Int, 12],
        }));
    }
}
MapOptions.__name__ = "MapOptions";
MapOptions.init_MapOptions();
export class GMapOptions extends MapOptions {
    constructor(attrs) {
        super(attrs);
    }
    static init_GMapOptions() {
        this.define(({ Boolean, Int, String }) => ({
            map_type: [String, "roadmap"],
            scale_control: [Boolean, false],
            styles: [String],
            tilt: [Int, 45],
        }));
    }
}
GMapOptions.__name__ = "GMapOptions";
GMapOptions.init_GMapOptions();
export class GMapPlot extends Plot {
    constructor(attrs) {
        super(attrs);
        this.use_map = true;
    }
    static init_GMapPlot() {
        this.prototype.default_view = GMapPlotView;
        // This seems to be necessary so that everything can initialize.
        // Feels very clumsy, but I'm not sure how the properties system wants
        // to handle something like this situation.
        this.define(({ String, Ref }) => ({
            map_options: [Ref(GMapOptions)],
            api_key: [String],
            api_version: [String, "3.43"],
        }));
        this.override({
            x_range: () => new Range1d(),
            y_range: () => new Range1d(),
        });
    }
}
GMapPlot.__name__ = "GMapPlot";
GMapPlot.init_GMapPlot();
//# sourceMappingURL=gmap_plot.js.map