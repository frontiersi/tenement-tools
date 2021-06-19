import { View } from "./view";
export declare class DOMView extends View {
    tagName: keyof HTMLElementTagNameMap;
    el: HTMLElement;
    /** @override */
    readonly root: DOMView;
    initialize(): void;
    remove(): void;
    css_classes(): string[];
    render(): void;
    renderTo(element: HTMLElement): void;
    protected _createElement(): HTMLElement;
}
//# sourceMappingURL=dom_view.d.ts.map