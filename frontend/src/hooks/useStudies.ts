import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { StudiesService, type StudyPublic } from "@/api";

export const studyKeys = {
  all: ["studies"] as const,
  list: (limit: number, offset: number) =>
    [...studyKeys.all, "list", { limit, offset }] as const,
  detail: (studyId: string) => [...studyKeys.all, studyId] as const,
};

export function useUploadStudy() {
  const qc = useQueryClient();
  return useMutation<StudyPublic, unknown, File>({
    mutationFn: (file) =>
      StudiesService.uploadStudyApiV1StudiesPost({
        formData: { file: file as unknown as string },
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: studyKeys.all });
    },
  });
}

export function useStudy(studyId: string | undefined) {
  return useQuery<StudyPublic>({
    queryKey: studyId ? studyKeys.detail(studyId) : studyKeys.all,
    queryFn: () => StudiesService.getStudyApiV1StudiesStudyIdGet({ studyId: studyId! }),
    enabled: Boolean(studyId),
  });
}

export function useStudiesList(limit = 100, offset = 0) {
  return useQuery<StudyPublic[]>({
    queryKey: studyKeys.list(limit, offset),
    queryFn: () => StudiesService.listStudiesApiV1StudiesGet({ limit, offset }),
  });
}
