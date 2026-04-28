import React from 'react';

interface ChartCardProps {
  title?: string;
  children: React.ReactNode;
  className?: string;
}

const ChartCard = ({ title, children, className = "" }: ChartCardProps) => (
  <div className={`
    rounded-xl border border-white/10
    bg-white/5 backdrop-blur-sm p-6
    hover:border-white/20 transition-all
    flex flex-col h-full
    ${className}
  `}>
    {title && (
      <h3 className="text-base font-semibold text-foreground mb-4">
        {title}
      </h3>
    )}
    <div className="flex-1 w-full min-h-[300px]">
      {children}
    </div>
  </div>
);

export default ChartCard;
