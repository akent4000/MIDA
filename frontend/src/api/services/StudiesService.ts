/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_upload_study_api_v1_studies_post } from '../models/Body_upload_study_api_v1_studies_post';
import type { StudyPublic } from '../models/StudyPublic';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class StudiesService {
    /**
     * Upload Study
     * @returns StudyPublic Successful Response
     * @throws ApiError
     */
    public static uploadStudyApiV1StudiesPost({
        formData,
    }: {
        formData: Body_upload_study_api_v1_studies_post,
    }): CancelablePromise<StudyPublic> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/v1/studies',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Study
     * @returns StudyPublic Successful Response
     * @throws ApiError
     */
    public static getStudyApiV1StudiesStudyIdGet({
        studyId,
    }: {
        studyId: string,
    }): CancelablePromise<StudyPublic> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/studies/{study_id}',
            path: {
                'study_id': studyId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Study Image
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getStudyImageApiV1StudiesStudyIdImageGet({
        studyId,
    }: {
        studyId: string,
    }): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/v1/studies/{study_id}/image',
            path: {
                'study_id': studyId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
}
