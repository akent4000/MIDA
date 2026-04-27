/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { InferenceRequest } from '../models/InferenceRequest';
import type { InferenceResultPublic } from '../models/InferenceResultPublic';
import type { InferenceSubmitResponse } from '../models/InferenceSubmitResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class InferenceService {
    /**
     * Submit Inference
     * @returns InferenceSubmitResponse Successful Response
     * @throws ApiError
     */
    public static submitInferenceApiV1StudiesStudyIdInferencePost({
        studyId,
        requestBody,
    }: {
        studyId: string,
        requestBody: InferenceRequest,
    }): CancelablePromise<InferenceSubmitResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/v1/studies/{study_id}/inference',
            path: {
                'study_id': studyId,
            },
            body: requestBody,
            mediaType: 'application/json',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Inference Result
     * @returns InferenceResultPublic Successful Response
     * @throws ApiError
     */
    public static getInferenceResultApiV1InferenceInferenceIdResultGet({
        inferenceId,
    }: {
        inferenceId: string,
    }): CancelablePromise<InferenceResultPublic> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/inference/{inference_id}/result',
            path: {
                'inference_id': inferenceId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Explanation
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getExplanationApiV1InferenceInferenceIdExplanationGet({
        inferenceId,
    }: {
        inferenceId: string,
    }): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/inference/{inference_id}/explanation',
            path: {
                'inference_id': inferenceId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
