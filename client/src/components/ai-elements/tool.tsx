"use client";

import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  ChevronDownIcon,
  CircleIcon,
  ClockIcon,
  WrenchIcon,
  XCircleIcon,
} from "lucide-react";
import type { ComponentProps, ReactNode } from "react";
import { CodeBlock } from "./code-block";

const MAX_STRING_LENGTH = 500;
const MAX_ARRAY_LENGTH = 20;
const MAX_OBJECT_KEYS = 20;
const MAX_DEPTH = 4;
const MAX_TOTAL_LENGTH = 20_000;

const formatForDisplay = (value: unknown, seen = new WeakSet<object>(), depth = 0): unknown => {
  if (value === null || typeof value === "undefined") {
    return value;
  }

  if (typeof value === "string") {
    return value.length > MAX_STRING_LENGTH
      ? `${value.slice(0, MAX_STRING_LENGTH)}…`
      : value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }

  if (typeof value === "bigint") {
    return value.toString();
  }

  if (typeof value === "function") {
    return `[Function ${value.name || "anonymous"}]`;
  }

  if (value instanceof Date) {
    return value.toISOString();
  }

  if (value instanceof Map) {
    return {
      type: "Map",
      size: value.size,
      entries: Array.from(value.entries())
        .slice(0, MAX_ARRAY_LENGTH)
        .map(([key, val]) => [formatForDisplay(key, seen, depth + 1), formatForDisplay(val, seen, depth + 1)]),
    };
  }

  if (value instanceof Set) {
    return {
      type: "Set",
      size: value.size,
      values: Array.from(value.values())
        .slice(0, MAX_ARRAY_LENGTH)
        .map((item) => formatForDisplay(item, seen, depth + 1)),
    };
  }

  if (typeof value === "object") {
    if (seen.has(value as object)) {
      return "[Circular]";
    }

    if (depth >= MAX_DEPTH) {
      const baseLabel = Array.isArray(value) ? "[Array]" : `[Object ${(value as { constructor?: { name?: string } }).constructor?.name || "Anonymous"}]`;
      return `${baseLabel} (depth limit reached)`;
    }

    seen.add(value as object);

    if (Array.isArray(value)) {
      const formatted = value
        .slice(0, MAX_ARRAY_LENGTH)
        .map((item) => formatForDisplay(item, seen, depth + 1));
      if (value.length > MAX_ARRAY_LENGTH) {
        formatted.push(`… ${value.length - MAX_ARRAY_LENGTH} more items`);
      }
      return formatted;
    }

    const entries = Object.entries(value as Record<string, unknown>);
    const limitedEntries = entries.slice(0, MAX_OBJECT_KEYS);
    const formattedEntries = Object.fromEntries(
      limitedEntries.map(([key, val]) => [key, formatForDisplay(val, seen, depth + 1)])
    );

    if (entries.length > MAX_OBJECT_KEYS) {
      formattedEntries.__truncated__ = `… ${entries.length - MAX_OBJECT_KEYS} more keys`;
    }

    return formattedEntries;
  }

  return String(value);
};

const safeStringify = (value: unknown, space = 2) => {
  try {
    const formatted = formatForDisplay(value);
    let json = JSON.stringify(formatted, null, space);
    if (json.length > MAX_TOTAL_LENGTH) {
      json = `${json.slice(0, MAX_TOTAL_LENGTH)}…`;
    }
    return json;
  } catch (error) {
    return JSON.stringify({ message: "Unable to display value", error: String(error) });
  }
};

export type ToolProps = ComponentProps<typeof Collapsible>;

export const Tool = ({ className, ...props }: ToolProps) => (
  <Collapsible
    className={cn("not-prose mb-4 w-full rounded-md border", className)}
    {...props}
  />
);

export type ToolHeaderProps = {
  title?: string;
  type: ToolUIPart["type"];
  state: ToolUIPart["state"];
  className?: string;
};

const getStatusBadge = (status: ToolUIPart["state"]) => {
  const labels = {
    "input-streaming": "Pending",
    "input-available": "Running",
    "output-available": "Completed",
    "output-error": "Error",
  } as const;

  const icons = {
    "input-streaming": <CircleIcon className="size-4" />,
    "input-available": <ClockIcon className="size-4 animate-pulse" />,
    "output-available": <CheckCircleIcon className="size-4 text-green-600" />,
    "output-error": <XCircleIcon className="size-4 text-red-600" />,
  } as const;

  return (
    <Badge className="gap-1.5 rounded-full text-xs" variant="secondary">
      {icons[status]}
      {labels[status]}
    </Badge>
  );
};

export const ToolHeader = ({
  className,
  title,
  type,
  state,
  ...props
}: ToolHeaderProps) => (
  <CollapsibleTrigger
    className={cn(
      "flex w-full items-center justify-between gap-4 p-3",
      className
    )}
    {...props}
  >
    <div className="flex items-center gap-2">
      <WrenchIcon className="size-4 text-muted-foreground" />
      <span className="font-medium text-sm">
        {title ?? type.split("-").slice(1).join("-")}
      </span>
      {getStatusBadge(state)}
    </div>
    <ChevronDownIcon className="size-4 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
  </CollapsibleTrigger>
);

export type ToolContentProps = ComponentProps<typeof CollapsibleContent>;

export const ToolContent = ({ className, ...props }: ToolContentProps) => (
  <CollapsibleContent
    className={cn(
      "data-[state=closed]:fade-out-0 data-[state=closed]:slide-out-to-top-2 data-[state=open]:slide-in-from-top-2 text-popover-foreground outline-none data-[state=closed]:animate-out data-[state=open]:animate-in",
      className
    )}
    {...props}
  />
);

export type ToolInputProps = ComponentProps<"div"> & {
  input: ToolUIPart["input"];
};

export const ToolInput = ({ className, input, ...props }: ToolInputProps) => (
  <div className={cn("space-y-2 overflow-hidden p-4", className)} {...props}>
    <h4 className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
      Parameters
    </h4>
    <div className="rounded-md bg-muted/50">
      <CodeBlock code={safeStringify(input)} language="json" />
    </div>
  </div>
);

export type ToolOutputProps = ComponentProps<"div"> & {
  output: ToolUIPart["output"];
  errorText: ToolUIPart["errorText"];
};

export const ToolOutput = ({
  className,
  output,
  errorText,
  ...props
}: ToolOutputProps) => {
  if (!(output || errorText)) {
    return null;
  }

  let Output = <div>{output as ReactNode}</div>;

  if (typeof output === "object") {
    Output = <CodeBlock code={safeStringify(output)} language="json" />;
  } else if (typeof output === "string") {
    Output = <CodeBlock code={output} language="json" />;
  }

  return (
    <div className={cn("space-y-2 p-4", className)} {...props}>
      <h4 className="font-medium text-muted-foreground text-xs uppercase tracking-wide">
        {errorText ? "Error" : "Result"}
      </h4>
      <div
        className={cn(
          "overflow-x-auto rounded-md text-xs [&_table]:w-full",
          errorText
            ? "bg-destructive/10 text-destructive"
            : "bg-muted/50 text-foreground"
        )}
      >
        {errorText && <div>{errorText}</div>}
        {Output}
      </div>
    </div>
  );
};
