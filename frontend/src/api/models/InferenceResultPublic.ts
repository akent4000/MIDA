/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export type InferenceResultPublic = {
    id: string;
    study_id: string;
    tool_id: string;
    task_id: string;
    status: string;
    result?: (Record<string, any> | null);
    gradcam_key?: (string | null);
    error_message?: (string | null);
    created_at: string;
    completed_at?: (string | null);
};

