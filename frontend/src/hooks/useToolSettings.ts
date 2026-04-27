import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ToolsService,
  type ToolSettingsConfig,
  type ToolSettingsPatch,
} from "@/api";

export const toolSettingsKeys = {
  all: ["tool-settings"] as const,
  detail: (toolId: string) => [...toolSettingsKeys.all, toolId] as const,
};

export function useToolConfig(toolId: string | undefined) {
  return useQuery<ToolSettingsConfig>({
    queryKey: toolId ? toolSettingsKeys.detail(toolId) : toolSettingsKeys.all,
    queryFn: () =>
      ToolsService.getToolConfigApiV1ToolsToolIdConfigGet({ toolId: toolId! }),
    enabled: Boolean(toolId),
  });
}

export function usePatchToolConfig(toolId: string) {
  const qc = useQueryClient();
  return useMutation<ToolSettingsConfig, unknown, ToolSettingsPatch>({
    mutationFn: (body) =>
      ToolsService.patchToolConfigApiV1ToolsToolIdConfigPatch({
        toolId,
        requestBody: body,
      }),
    onSuccess: (data) => {
      qc.setQueryData(toolSettingsKeys.detail(toolId), data);
    },
  });
}
