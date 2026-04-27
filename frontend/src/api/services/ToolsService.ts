/* hand-written — replace on next openapi-typescript-codegen run */
/* tslint:disable */
/* eslint-disable */
import type { ToolSettingsConfig, ToolSettingsPatch } from '../models/ToolSettingsConfig';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';

export class ToolsService {
    /**
     * Get Tool Config
     * @returns ToolSettingsConfig Successful Response
     * @throws ApiError
     */
    public static getToolConfigApiV1ToolsToolIdConfigGet({
        toolId,
    }: {
        toolId: string,
    }): CancelablePromise<ToolSettingsConfig> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/tools/{tool_id}/config',
            path: { 'tool_id': toolId },
            errors: { 404: `Tool not found`, 422: `Validation Error` },
        });
    }

    /**
     * Patch Tool Config
     * @returns ToolSettingsConfig Successful Response
     * @throws ApiError
     */
    public static patchToolConfigApiV1ToolsToolIdConfigPatch({
        toolId,
        requestBody,
    }: {
        toolId: string,
        requestBody: ToolSettingsPatch,
    }): CancelablePromise<ToolSettingsConfig> {
        return __request(OpenAPI, {
            method: 'PATCH',
            url: '/api/v1/tools/{tool_id}/config',
            path: { 'tool_id': toolId },
            body: requestBody,
            mediaType: 'application/json',
            errors: { 400: `Tool has no settings`, 404: `Tool not found`, 422: `Validation Error` },
        });
    }
}
