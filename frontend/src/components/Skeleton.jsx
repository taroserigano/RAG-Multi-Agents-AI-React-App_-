/**
 * Loading skeleton components for better perceived performance.
 */

export function Skeleton({ className = "" }) {
  return (
    <div
      className={`animate-pulse bg-[var(--bg-secondary)]/60 rounded-lg ${className}`}
    />
  );
}

export function DocumentSkeleton() {
  return (
    <div className="p-4 rounded-xl bg-[var(--bg-secondary)]/40 border border-[var(--border-subtle)]">
      <div className="flex items-center gap-3">
        <Skeleton className="h-10 w-10 rounded-lg" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
        </div>
        <Skeleton className="h-8 w-8 rounded-lg" />
      </div>
    </div>
  );
}

export function DocumentListSkeleton({ count = 5 }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <DocumentSkeleton key={i} />
      ))}
    </div>
  );
}

export function ImageSkeleton() {
  return (
    <div className="aspect-square rounded-xl bg-[var(--bg-secondary)]/40 border border-[var(--border-subtle)] overflow-hidden">
      <Skeleton className="h-full w-full rounded-none" />
    </div>
  );
}

export function ImageGallerySkeleton({ count = 6 }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <ImageSkeleton key={i} />
      ))}
    </div>
  );
}

export function ChatSkeleton() {
  return (
    <div className="space-y-6">
      {/* User message skeleton */}
      <div className="flex items-start gap-3 justify-end">
        <div className="max-w-[80%] space-y-2">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-32" />
        </div>
        <Skeleton className="h-8 w-8 rounded-xl" />
      </div>

      {/* Assistant message skeleton */}
      <div className="flex items-start gap-3">
        <Skeleton className="h-8 w-8 rounded-xl" />
        <div className="flex-1 max-w-[85%] space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      </div>
    </div>
  );
}

export function StatCardSkeleton() {
  return (
    <div className="p-4 rounded-xl bg-[var(--bg-secondary)]/40 border border-[var(--border-subtle)]">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-8 w-12" />
        </div>
        <Skeleton className="h-10 w-10 rounded-lg" />
      </div>
    </div>
  );
}

export default Skeleton;
