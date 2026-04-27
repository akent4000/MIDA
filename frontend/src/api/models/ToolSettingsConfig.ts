/* hand-written — replace on next openapi-typescript-codegen run */
/* tslint:disable */
/* eslint-disable */

export type SettingType = "select" | "toggle" | "number";

export type SettingOption = {
    value: string;
    label: string;
    description?: string;
};

export type SettingField = {
    key: string;
    label: string;
    type: SettingType;
    default: unknown;
    description?: string;
    options?: Array<SettingOption>;
    min?: number;
    max?: number;
    step?: number;
};

export type ToolSettingsConfig = {
    tool_id: string;
    schema: Array<SettingField>;
    values: Record<string, unknown>;
};

export type ToolSettingsPatch = {
    values: Record<string, unknown>;
};
