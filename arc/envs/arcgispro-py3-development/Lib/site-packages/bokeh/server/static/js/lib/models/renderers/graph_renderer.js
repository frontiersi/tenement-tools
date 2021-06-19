import { DataRenderer, DataRendererView } from "./data_renderer";
import { GlyphRenderer } from "./glyph_renderer";
import { Expression } from "../expressions/expression";
import { LayoutProvider } from "../graphs/layout_provider";
import { GraphHitTestPolicy, NodesOnly } from "../graphs/graph_hit_test_policy";
import { build_view } from "../../core/build_views";
import { XYGlyph } from "../glyphs/xy_glyph";
import { MultiLine } from "../glyphs/multi_line";
import { Patches } from "../glyphs/patches";
import { assert } from "../../core/util/assert";
export class GraphRendererView extends DataRendererView {
    get glyph_view() {
        return this.node_view.glyph;
    }
    async lazy_initialize() {
        await super.lazy_initialize();
        const graph = this.model;
        // TODO: replace this with bi-variate transforms
        let xs_ys = null;
        let x_y = null;
        const xs_expr = new class extends Expression {
            _v_compute(source) {
                assert(xs_ys == null);
                const [xs] = xs_ys = graph.layout_provider.get_edge_coordinates(source);
                return xs;
            }
        };
        const ys_expr = new class extends Expression {
            _v_compute(_source) {
                assert(xs_ys != null);
                const [, ys] = xs_ys;
                xs_ys = null;
                return ys;
            }
        };
        const x_expr = new class extends Expression {
            _v_compute(source) {
                assert(x_y == null);
                const [x] = x_y = graph.layout_provider.get_node_coordinates(source);
                return x;
            }
        };
        const y_expr = new class extends Expression {
            _v_compute(_source) {
                assert(x_y != null);
                const [, y] = x_y;
                x_y = null;
                return y;
            }
        };
        const { edge_renderer, node_renderer } = this.model;
        // TODO: XsYsGlyph or something
        if (!(edge_renderer.glyph instanceof MultiLine || edge_renderer.glyph instanceof Patches)) {
            throw new Error(`${this}.edge_renderer.glyph must be a MultiLine glyph`);
        }
        if (!(node_renderer.glyph instanceof XYGlyph)) {
            throw new Error(`${this}.node_renderer.glyph must be a XYGlyph glyph`);
        }
        edge_renderer.glyph.properties.xs.internal = true;
        edge_renderer.glyph.properties.ys.internal = true;
        node_renderer.glyph.properties.x.internal = true;
        node_renderer.glyph.properties.y.internal = true;
        edge_renderer.glyph.xs = { expr: xs_expr };
        edge_renderer.glyph.ys = { expr: ys_expr };
        node_renderer.glyph.x = { expr: x_expr };
        node_renderer.glyph.y = { expr: y_expr };
        const { parent } = this;
        this.edge_view = await build_view(edge_renderer, { parent });
        this.node_view = await build_view(node_renderer, { parent });
    }
    connect_signals() {
        super.connect_signals();
        this.connect(this.model.layout_provider.change, () => {
            this.edge_view.set_data();
            this.node_view.set_data();
            this.request_render();
        });
    }
    remove() {
        this.edge_view.remove();
        this.node_view.remove();
        super.remove();
    }
    _render() {
        this.edge_view.render();
        this.node_view.render();
    }
    renderer_view(renderer) {
        if (renderer instanceof GlyphRenderer) {
            if (renderer == this.edge_view.model)
                return this.edge_view;
            if (renderer == this.node_view.model)
                return this.node_view;
        }
        return super.renderer_view(renderer);
    }
}
GraphRendererView.__name__ = "GraphRendererView";
export class GraphRenderer extends DataRenderer {
    constructor(attrs) {
        super(attrs);
    }
    static init_GraphRenderer() {
        this.prototype.default_view = GraphRendererView;
        this.define(({ Ref }) => ({
            layout_provider: [Ref(LayoutProvider)],
            node_renderer: [Ref(GlyphRenderer)],
            edge_renderer: [Ref(GlyphRenderer)],
            selection_policy: [Ref(GraphHitTestPolicy), () => new NodesOnly()],
            inspection_policy: [Ref(GraphHitTestPolicy), () => new NodesOnly()],
        }));
    }
    get_selection_manager() {
        return this.node_renderer.data_source.selection_manager;
    }
}
GraphRenderer.__name__ = "GraphRenderer";
GraphRenderer.init_GraphRenderer();
//# sourceMappingURL=graph_renderer.js.map