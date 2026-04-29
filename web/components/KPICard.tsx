import React from 'react';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: string;
}

/**
 * Format a numeric value as Indian Rupee with smart scaling.
 * Handles both raw numbers and pre-formatted strings from the backend.
 */
function formatINR(value: string | number): string {
  if (typeof value === 'number') {
    if (value >= 1_00_00_000) return `₹${(value / 1_00_00_000).toFixed(2)} Cr`;
    if (value >= 1_00_000)    return `₹${(value / 1_00_000).toFixed(2)} L`;
    if (value >= 1_000)       return `₹${(value / 1_000).toFixed(1)}K`;
    return `₹${value.toLocaleString('en-IN')}`;
  }
  // String pass-through: swap any leading $ → ₹, leave ₹ and non-currency as-is
  return value.replace(/^-?\$/, (m) => m.startsWith('-') ? '-₹' : '₹');
}

const KPICard = ({ title, value, subtitle, icon, trend }: KPICardProps) => (
  <div className="
    rounded-xl border border-white/10 
    bg-white/5 backdrop-blur-sm
    p-6 flex flex-col gap-2
    hover:bg-white/8 transition-all
    hover:border-white/20
    hover:shadow-lg hover:shadow-purple-500/10
    group select-none
  ">
    <div className="flex items-center justify-between">
      <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
        {title}
      </span>
      {icon && (
        <span className="text-2xl opacity-60 group-hover:opacity-100 transition-opacity">
          {icon}
        </span>
      )}
    </div>
    <div className="text-3xl font-bold text-foreground tracking-tight select-none">
      {formatINR(value)}
    </div>
    {subtitle && (
      <p className="text-xs text-muted-foreground">
        {subtitle}
      </p>
    )}
    {trend && (
      <p className={`text-xs font-medium ${
        trend.startsWith('+') 
          ? 'text-green-400' 
          : trend === '0%' 
            ? 'text-muted-foreground'
            : 'text-red-400'
      }`}>
        {trend}
      </p>
    )}
  </div>
);

export default KPICard;
