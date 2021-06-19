import { ColumnarDataSource } from "../sources/columnar_data_source";
import { Model } from "../../model";
import { Arrayable } from "../../core/types";
import * as p from "../../core/properties";
export declare namespace Expression {
    type Attrs = p.AttrsOf<Props>;
    type Props = Model.Props;
}
export interface Expression extends Expression.Attrs {
}
export declare abstract class Expression extends Model {
    properties: Expression.Props;
    constructor(attrs?: Partial<Expression.Attrs>);
    protected _connected: Set<ColumnarDataSource>;
    protected _result: Map<ColumnarDataSource, Arrayable>;
    initialize(): void;
    protected abstract _v_compute(source: ColumnarDataSource): Arrayable;
    v_compute(source: ColumnarDataSource): Arrayable;
}
export declare namespace ScalarExpression {
    type Attrs = p.AttrsOf<Props>;
    type Props = Model.Props;
}
export interface ScalarExpression<T> extends ScalarExpression.Attrs {
}
export declare abstract class ScalarExpression<T> extends Model {
    properties: ScalarExpression.Props;
    constructor(attrs?: Partial<ScalarExpression.Attrs>);
    protected _connected: Set<ColumnarDataSource>;
    protected _result: Map<ColumnarDataSource, T>;
    initialize(): void;
    protected abstract _compute(source: ColumnarDataSource): T;
    compute(source: ColumnarDataSource): T;
}
//# sourceMappingURL=expression.d.ts.map