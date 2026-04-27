/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ModelListItem } from '../models/ModelListItem';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class ModelsService {
    /**
     * List Models
     * @returns ModelListItem Successful Response
     * @throws ApiError
     */
    public static listModelsApiV1ModelsGet(): CancelablePromise<Array<ModelListItem>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/models',
        });
    }
}
